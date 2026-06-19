"""Sketched LoRA (SVD-sketching) continual learner for LAMDA-PILOT.

A *single* fixed-rank-r̂ "sketch" adapter B̂Â summarises the history of all past
tasks.  Unlike SeqLoRA (one adapter that drifts) it is never trained in place:
each task trains a fresh residual on top of the *frozen* sketch, and after the
task the (sketch ⊕ residual) sum is re-compressed back to rank r̂ via randomized
SVD.  Memory is therefore bounded at rank r̂ no matter how many tasks arrive.

This port fixes the period to **P = 1** (compress after every task -- the regime
with no cross-task interference blind spots).  The sketch rank ``svd_rank`` (r̂)
defaults to the per-task adapter rank but may be set larger (slot 0 is then
resized to a rank-r̂ factorisation) to probe whether more sketch capacity cures
old-task eviction.  ``n_lora_blocks`` optionally restricts LoRA to the first n
transformer blocks.  Every other hyperparameter is inherited from
``models/lora.py``.

Slot mapping onto the shared LoRA scaffold (backbone/vit_lora.py)
----------------------------------------------------------------
We reuse the existing per-task LoRA slots -- *no backbone changes*:

  * slot 0  -> the frozen sketch  B̂Â   (rank r̂ = svd_rank, resized if r̂ != r)
  * slot 1  -> the trainable current-task residual  B_new A_new  (rank r)

Training task t (inherited ``incremental_train``/``_train``):
  * ``freeze_to_task(1)`` keeps only slot 1 trainable; slot 0 stays frozen.
  * forward routes ``task=1, merge=True`` -> sums slots {0,1} =
        W·x + s·(B̂Â)·x + s·(B_new A_new)·x      (s = lora_scaling)
  * task-local cross-entropy (inherited).

After training (``_compress``, called at the end of ``_train`` so eval -- which
runs before ``after_task`` -- sees the compressed state):
  * ΔW = B̂Â + B_new A_new   (unscaled factor products, per layer, per q/v proj)
  * B̂, Â = rand_svd(ΔW, r̂, oversampling)  -> written back into slot 0
  * slot 1 is reset (A kaiming, B zero) -> a clean residual + a no-op at eval.

Inference (single shared sketch, like SeqLoRA but compressed):
  * CIL -> forward(x) routes default_task=1, merge=True; residual B is zero, so
           the result is W·x + s·B̂Â·x.
  * TIL -> route to slot 0 (``_eval_adapter`` -> 0), merge=False -> W·x + s·B̂Â·x,
           masked to the known task's class slice.
"""

import json
import logging
import math
import os
import sys

import torch
from torch import nn

from models.lora import Learner as LoRALearner

# trusted randomized-SVD implementation (vendored into utils/ for self-containment)
from utils.randsvd import rand_svd

# fixed-slot convention: 0 = frozen sketch, 1 = trainable residual
SKETCH = 0
RESIDUAL = 1


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # P = 1.  r̂ (sketch rank) may differ from the residual/adapter rank.
        self.lora_rank = args.get("lora_rank", 10)        # per-task residual rank r
        self.svd_rank = args.get("svd_rank", self.lora_rank)   # sketch target rank r̂
        self.oversampling = args.get("svd_oversampling", 10)
        # optional depth restriction: LoRA only on the first n blocks (else all)
        self.n_lora_blocks = args.get("n_lora_blocks", None)
        # train on sketch(0)+residual(1); both eval paths reduce to the sketch
        self.train_merge = True
        self._network.merge = True
        # if r̂ != r, the sketch slot must hold a rank-r̂ factorisation
        if self.svd_rank != self.lora_rank:
            self._resize_sketch_slot()
        # -- compression diagnostics (test Corollary 3's structural assumption) --
        # records, per compression event, the singular spectrum of the
        # accumulated delta_W so we can read sigma_{r̂+1} and the retained-energy
        # fraction directly off each truncation.  See Remark 2 condition (iii).
        self.sketch_diag = bool(args.get("sketch_diag", True))
        self._diag_records = []
        seed = args["seed"] if not isinstance(args.get("seed"), list) else args["seed"][0]
        tag = "r{}_b{}".format(self.svd_rank, self.n_lora_blocks or "all")
        self._diag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run_logs", "sketchlora_diag_{}_seed{}.json".format(tag, seed))

    # -- which attention blocks carry LoRA (all, or the first n) --------
    def _all_attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    def _active_attns(self):
        attns = self._all_attns()
        return attns if self.n_lora_blocks is None else attns[:self.n_lora_blocks]

    def _resize_sketch_slot(self):
        """Replace slot-0 (the frozen sketch) with rank-r̂ Linears on active
        blocks, zero-initialised (B̂Â = 0 until the first compression)."""
        for attn in self._active_attns():
            dim = attn.dim
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                ref = A_list[SKETCH].weight
                newA = nn.Linear(dim, self.svd_rank, bias=False)
                newB = nn.Linear(self.svd_rank, dim, bias=False)
                nn.init.zeros_(newA.weight)
                nn.init.zeros_(newB.weight)
                newA.to(ref.device, ref.dtype)
                newB.to(ref.device, ref.dtype)
                for p in list(newA.parameters()) + list(newB.parameters()):
                    p.requires_grad = False
                A_list[SKETCH] = newA
                B_list[SKETCH] = newB

    def _freeze_inactive_blocks(self):
        """When depth-restricted, keep the residual frozen on blocks >= n so the
        optimiser never touches them (they stay zero -> no contribution)."""
        if self.n_lora_blocks is None:
            return
        for attn in self._all_attns()[self.n_lora_blocks:]:
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for p in mlist[RESIDUAL].parameters():
                    p.requires_grad = False

    # -- adapter routing (override the lora.Learner indirection) --------
    def _train_adapter(self):
        return RESIDUAL          # train the residual on top of the frozen sketch

    def _eval_adapter(self, task):
        return SKETCH            # TIL routes to the single compressed sketch

    # -- train then compress (eval runs before after_task) --------------
    def _train(self, train_loader):
        self._freeze_inactive_blocks()   # re-freeze before the optimiser is built
        super()._train(train_loader)
        self._compress()

    @torch.no_grad()
    def _compress(self):
        """RandSVD-compress (sketch ⊕ residual) -> sketch, per layer & proj."""
        retained, sigma_next, fro = [], [], []   # per (layer, proj) diagnostics
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_s, B_s = A_list[SKETCH].weight, B_list[SKETCH].weight     # [r,d],[d,r]
                A_r, B_r = A_list[RESIDUAL].weight, B_list[RESIDUAL].weight
                delta_W = B_s @ A_s + B_r @ A_r                            # [d, d], unscaled
                if self.sketch_diag:
                    S = torch.linalg.svdvals(delta_W.float())             # full spectrum, desc
                    energy = S.pow(2)
                    total = energy.sum()
                    r = self.svd_rank
                    retained.append((energy[:r].sum() / total).item() if total > 0 else 1.0)
                    sigma_next.append(S[r].item() if S.numel() > r else 0.0)
                    fro.append(total.sqrt().item())
                B_hat, A_hat = rand_svd(delta_W, self.svd_rank, self.oversampling)
                B_s.data.copy_(B_hat.to(B_s.device, B_s.dtype))
                A_s.data.copy_(A_hat.to(A_s.device, A_s.dtype))
                # reset the residual: kaiming A, zero B -> clean + eval no-op
                nn.init.kaiming_uniform_(A_r, a=math.sqrt(5))
                nn.init.zeros_(B_r)
        if self.sketch_diag:
            self._record_diag(retained, sigma_next, fro)

    def _record_diag(self, retained, sigma_next, fro):
        """Aggregate + persist the per-compression singular-spectrum stats."""
        import numpy as np
        rec = {
            "task": self._cur_task,
            "retained_energy": retained,        # frac of ||ΔW||² kept by top-r̂
            "sigma_next": sigma_next,           # σ_{r̂+1}(ΔW), per (layer,proj)
            "fro": fro,                         # ||ΔW||_F, per (layer,proj)
            "retained_mean": float(np.mean(retained)),
            "retained_min": float(np.min(retained)),
            "fro_mean": float(np.mean(fro)),
        }
        self._diag_records.append(rec)
        os.makedirs(os.path.dirname(self._diag_path), exist_ok=True)
        with open(self._diag_path, "w") as f:
            json.dump(self._diag_records, f, indent=2)
        logging.info(
            "[SketchDiag] task {}: retained-energy mean={:.3f} min={:.3f} | "
            "||ΔW||_F mean={:.3f}".format(
                self._cur_task, rec["retained_mean"], rec["retained_min"], rec["fro_mean"]))
