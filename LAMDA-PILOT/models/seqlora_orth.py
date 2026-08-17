"""SeqLoRA + O-LoRA's orthogonality penalty against a FROZEN snapshot of its
own single adapter -- a mechanism-isolation ablation, not a method.

Motivation (2026-08-17). In the bankcap_wave1_imagenetr20t campaign, O-LoRA
constrained to a 3-4MB bank ends up with exactly TWO LoRA slots (slot 0 frozen
after task 0, slot 1 trained continuously for the remaining 19 tasks -- see
models/olora.py::_train_adapter and docs/bounded_bank_memory_changelog.md) and
still beats SeqLoRA by ~11.6 top5 / ~21 points of Forgetting on ImageNet-R-20t.
Two things differ between "O-LoRA capped to 2 slots" and SeqLoRA, and the
campaign cannot tell them apart:

  D1 (additive memory): the frozen slot 0's delta is summed into every forward
      (train_merge=True), so the deployed function permanently contains a
      task-0-specialised component.
  D2 (regulariser): the live slot's A is penalised toward orthogonality with
      the frozen A_0 every step, with lamda_1=0.5.

The evidence already on record points at D2, not D1: the retention gain in the
final per-class-block accuracies is spread UNIFORMLY over all 19 old tasks,
whereas slot 0 only ever encoded task 0 -- an additive-memory effect would be
localised to the 00-09 block. This file supplies the direct test of that
reading by isolating D2 with D1 removed entirely.

Mechanism: train the single pinned slot 0 exactly as SeqLoRA does (no second
slot, no merge, no re-initialisation, ever), but at the END of task
`orth_ref_task` (default 0) take a detached snapshot of every wrapped module's
A_q/A_v and, from the next task onward, add O-LoRA's penalty against that
FROZEN snapshot:

    loss = CE  +  lamda_1 * sum_{blocks,proj} | A_live @ A_ref^T |.sum()
               +  lamda_2 * || live slot ||_2

The snapshot is taken once and never updated -- deliberately the most stale
possible reference, and O(1) in task count (one [r, dim] matrix per (block,
proj) pair, 0.703MB total at rank=10, counted in persistent_state below). The
penalty is numerically the same expression models/olora.py::_orth_and_l2
computes; with a single reference slot its `A_prev` cache is exactly this
snapshot, so `torch.abs(A_t @ A_ref.t()).sum()` is that method's identical
per-(block,proj) term, not an approximation of it.

Companion arm of this ablation: O-LoRA at the same bank caps with lamda_1=0
(D1 without D2), which needs no code -- just the config field. See
scripts/olora_mechanism_ablation_imagenetr20t.slurm.

Expected outcome if the D2 reading is right: this learner lands near capped
O-LoRA (~77 top5) at a CONSTANT 2.70MB, and the lamda_1=0 arm falls back
toward SeqLoRA (~66 top5). If instead the lamda_1=0 arm holds up and this one
collapses, the additive frozen slot was doing the work and the D2 reading is
wrong.
"""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.seqlora import Learner as SeqLoRALearner
from utils.toolkit import tensor2numpy


class Learner(SeqLoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.lamda_1 = args.get("lamda_1", 0.5)   # orthogonality weight (O-LoRA's own default)
        self.lamda_2 = args.get("lamda_2", 0.0)   # L2 on the live slot
        # Which task's end-of-training A becomes the permanent reference. 0
        # matches what the capped O-LoRA run actually froze (its slot 0, frozen
        # once task 0 finished).
        self.orth_ref_task = args.get("orth_ref_task", 0)
        # list of (A_q_ref, A_v_ref) detached [r, dim] tensors, one entry per
        # wrapped attention module, in attn_modules() order. None until the
        # reference task finishes -- the penalty is inactive up to that point,
        # matching O-LoRA's own `if t > 0` guard.
        self._orth_ref = None

    def _attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    @torch.no_grad()
    def _snapshot_orth_ref(self):
        """Freeze the current A_q/A_v of the single live slot as the permanent
        orthogonality reference. Called exactly once, at the end of task
        `orth_ref_task`; never refreshed afterward (that staleness is the
        point of the ablation)."""
        t = self._train_adapter()   # always 0 -- SeqLoRA pins the single slot
        self._orth_ref = [(attn.lora_A_q[t].weight.detach().clone(),
                           attn.lora_A_v[t].weight.detach().clone())
                          for attn in self._attns()]

    def _orth_and_l2(self):
        """Same expression as models/olora.py::_orth_and_l2, with the frozen
        single-slot snapshot standing in for its concatenated `A_prev` cache.
        Returns (0.0, l2) before the snapshot exists."""
        t = self._train_adapter()
        orth = 0.0
        l2 = 0.0
        for idx, attn in enumerate(self._attns()):
            for j, (A_list, B_list) in enumerate(((attn.lora_A_q, attn.lora_B_q),
                                                  (attn.lora_A_v, attn.lora_B_v))):
                A_t = A_list[t].weight                       # [r, dim]
                l2 = l2 + torch.norm(A_t, p=2) + torch.norm(B_list[t].weight, p=2)
                if self._orth_ref is not None:
                    A_ref = self._orth_ref[idx][j]           # [r, dim] (frozen)
                    orth = orth + torch.abs(A_t @ A_ref.t()).sum()
        return orth, l2

    def _train(self, train_loader):
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, orth_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(),
                                       merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                orth, l2 = self._orth_and_l2()
                loss = ce + self.lamda_1 * orth + self.lamda_2 * l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                orth_run += float(orth)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, Orth {:.3f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    orth_run / len(train_loader), train_acc))
        logging.info("[SeqLoRA-Orth] Task {} done. Acc {:.2f}, final Orth {:.3f}".format(
            self._cur_task, train_acc, orth_run / len(train_loader)))

        # Snapshot AFTER this task's training, so the reference is the fully
        # trained A -- the same state O-LoRA's slot 0 is in when freeze_to_task
        # locks it at the start of task 1.
        if self._cur_task == self.orth_ref_task and self._orth_ref is None:
            self._snapshot_orth_ref()
            ref_mb = sum(a.numel() * a.element_size() + b.numel() * b.element_size()
                         for a, b in self._orth_ref) / 1024 / 1024
            logging.info("[SeqLoRA-Orth] froze orthogonality reference from task {} ({:.4f} MB)".format(
                self.orth_ref_task, ref_mb))

    def persistent_state(self):
        """models/lora.py's generic accounting (the single slot + head) plus the
        frozen reference, which is real resident memory held for the whole run
        and would otherwise go unreported -- the same gap O-LoRA's own
        `_orth_prev_{q,v}` cache has (see the bank-cap analysis, 2026-08-17).
        Constant in task count by construction."""
        base = super().persistent_state()
        ref_bytes = 0
        if self._orth_ref is not None:
            ref_bytes = sum(a.numel() * a.element_size() + b.numel() * b.element_size()
                            for a, b in self._orth_ref)
        total_bytes = base["bytes"] + ref_bytes
        breakdown = dict(base["breakdown"])
        breakdown["orth_reference"] = ref_bytes
        return {"params": int(total_bytes // 4), "bytes": int(total_bytes), "breakdown": breakdown}
