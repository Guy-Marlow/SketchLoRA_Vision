"""SLAO -- "Single LoRA continual learning via continual merging" (2026-10-27
user-requested port of "Merge before Forget," arXiv:2512.23017, Qiao &
Mahdavi). See the research report this implementation was built from:
scratchpad/analysis_merge_before_forget.md (verified against the paper's own
Algorithm 1 / Appendix B.11 text before any code was written).

ALGORITHM (paper's Algorithm 1, restated in this project's terms). SLAO keeps
exactly TWO LoRA slots per (block, q/v) module, never more:
  MERGE (slot 0) -- the single deployed/inference pair (A_merge, B_merge).
    Frozen forever (never receives a gradient) -- only ever updated by the
    closed-form boundary step below, never by backprop.
  FT (slot 1) -- the "fine-tune" pair (A_ft, B_ft) actually trained this task.

Per-task training is PLAIN task-local cross-entropy -- models/lora.py's base
_train() is reused completely unmodified; SLAO adds no loss term at all
(unlike O-LoRA's orthogonality penalty).

At the END of every task's training (boundary step, _slao_boundary() below),
for every module:
  1. A_merge := A_ft                         (plain overwrite, every task)
  2. B_merge := B_merge + beta(t)*(B_ft - B_merge)   (EMA-style asymmetric
     merge -- A and B are updated by genuinely different rules; this
     asymmetry, motivated by the paper's own empirical finding that A changes
     little across tasks while B carries most of the task-specific signal,
     is the method's central design idea)
  3. A_ft := QR((A_ft)^T)'s sign-corrected Q, transposed back to [r,dim]
     (orthogonal-basis re-init for the NEXT task's A_ft -- the only place
     orthogonality is enforced; no per-step penalty, unlike O-LoRA)
  4. B_ft is left COMPLETELY untouched -- next task's fine-tuning continues
     from wherever this task's B_ft ended ("Last-FT" init, confirmed the
     paper's own best-performing ablation, Table 2).
No special-casing is needed for the first task: with beta(0)=1 (see _beta
below) and B_merge starting at its construction-time zero-init, step 2
reduces to B_merge := B_ft exactly, matching the paper's own task-1 base
case (Algorithm 1 line 1) for free.

QR is the paper's own published default (and empirically best/tied-best vs.
SVD/randomized-SVD, Appendix B.6 Table 11) -- the only orthogonal-basis
method implemented here, per this being a MINIMAL port.

PERSISTENT MEMORY: inherited models/lora.py::persistent_state() already sums
every currently-allocated slot (here: MERGE + FT, fixed forever, lora_n_slots
must be 2) -- no override needed. This is the straightforward "4 live
matrices" accounting (A_merge, B_merge, A_ft, B_ft each a real, separately-
allocated nn.Parameter) rather than the paper's own idealized "3 matrices"
bound (Appendix B.11: "keep fine-tuned B_i, merged B_i, and fine-tuned A_i"
-- which only holds at the instant right after a merge, when A_merge==A_ft
numerically; the moment the next task's training resumes they diverge again)
-- see the research report's §2 for the full justification of this choice.
Either way this is O(1) in the number of tasks T, the paper's headline claim.

SCOPE: oracle (task-boundary) CIL path only, not the sample-boundary
streaming machinery -- the paper's Algorithm 1 is itself strictly per-task,
same scope restriction models/sketchlora_orthref.py / sketchlora_align.py
already carry for their own unsupported combinations.
"""
import logging

import torch

from models.lora import Learner as LoRALearner

MERGE = 0
FT = 1


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        assert self._network.backbone.n_tasks == 2, \
            "SLAO needs exactly 2 LoRA slots (MERGE + FT) -- set \"lora_n_slots\": 2 " \
            "in the config. Got n_tasks={}.".format(self._network.backbone.n_tasks)
        # beta(t) schedule for the B merge step (paper sec 4.1/Fig 4): "inverse_t"
        # (default, beta(t)=1/(t+1), 0-indexed self._cur_task -- the paper's own
        # published default and best/lowest-variance choice) or "const" (one of
        # the paper's own ablated fixed values, e.g. 0.1/0.5/0.9).
        self.beta_mode = args.get("slao_beta_mode", "inverse_t")
        assert self.beta_mode in ("inverse_t", "const")
        self.beta_const = args.get("slao_beta_const", 0.5)
        # SLAO's deployed/inference model is always the SINGLE merged pair
        # (MERGE slot alone) -- never a sum of slots the way O-LoRA/SketchLoRA's
        # merge=True forward works. Force this at construction rather than
        # relying on every config remembering "lora_merge": false.
        self._network.merge = False

    # -- beta(t) schedule ------------------------------------------------
    def _beta(self, cur_task):
        """cur_task is 0-indexed (self._cur_task); the paper's own t is
        1-indexed with beta(1)=1 as part of Algorithm 1's INITIALIZATION
        (line 1), not the i=2..T loop's schedule. Forcing beta=1.0 at
        cur_task==0 regardless of beta_mode reproduces that base case exactly
        under every schedule (not just "inverse_t", where it already falls
        out for free) -- without it, a fixed-constant beta_mode would
        incorrectly blend B_ft,1 with the zero-initialized B_merge,0 instead
        of hard-copying it (a real edge case the paper itself never
        discusses -- see the research report's open question #2)."""
        if cur_task == 0:
            return 1.0
        if self.beta_mode == "const":
            return self.beta_const
        return 1.0 / (cur_task + 1)

    # -- adapter routing (override the lora.Learner indirection) --------
    def _train_adapter(self):
        return FT   # always train the fine-tune slot; MERGE never gets a gradient

    def _eval_adapter(self, task):
        return MERGE   # TIL routes to the single merged pair too -- SLAO has no per-task routing

    def _deployed_forward(self, inputs):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=MERGE, merge=False)

    # -- boundary step: merge FT into MERGE, then re-init A_ft for the next task
    @torch.no_grad()
    def _slao_boundary(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        beta = self._beta(self._cur_task)
        ortho_errs, b_deltas = [], []
        for attn in net.attn_modules():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                A_merge, B_merge = A_list[MERGE].weight, B_list[MERGE].weight
                A_ft, B_ft = A_list[FT].weight, B_list[FT].weight

                b_deltas.append(torch.linalg.norm(B_ft - B_merge, ord="fro").item())

                # 1. A_merge := A_ft (plain overwrite)
                A_merge.data.copy_(A_ft.data)
                # 2. B_merge := B_merge + beta*(B_ft - B_merge)
                B_merge.data.add_(beta * (B_ft.data - B_merge.data))
                # 3. A_ft := orthogonal basis of the just-finished A_ft, for next task.
                #    QR((A_ft)^T): A_ft is [r,dim], so (A_ft)^T is [dim,r], Q is [dim,r]
                #    with orthonormal COLUMNS. Sign-correct each column by sign(diag(R))
                #    (removes QR's inherent +-1 column-sign ambiguity -> deterministic).
                Q, R = torch.linalg.qr(A_ft.data.t())
                signs = torch.sign(torch.diagonal(R))
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # avoid zeroing a column
                Q = Q * signs.unsqueeze(0)
                A_ft.data.copy_(Q.t())   # back to [r,dim], now with orthonormal ROWS
                ortho_errs.append(
                    torch.linalg.norm(A_ft.data @ A_ft.data.t() - torch.eye(
                        A_ft.shape[0], device=A_ft.device, dtype=A_ft.dtype), ord="fro").item())
                # 4. B_ft: left untouched (continues training from here next task)

        logging.info(
            "[SLAO] task {} boundary: beta={:.4f} | mean ||B_ft-B_merge||_F={:.4f} | "
            "mean ||A_ft A_ft^T - I||_F={:.2e} (orth re-init sanity check, should be ~0)".format(
                self._cur_task, beta, sum(b_deltas) / len(b_deltas), sum(ortho_errs) / len(ortho_errs)))

    def _train(self, train_loader):
        super()._train(train_loader)
        self._slao_boundary()
        net = self._network.module if hasattr(self._network, "module") else self._network
        # route the bare CIL evaluator (net(inputs), no explicit task) to the
        # just-updated MERGE slot -- incremental_train's own
        # `default_task = self._train_adapter()` (models/lora.py) pointed this
        # at FT before training started; reset it here, same convention
        # models/sketchlora.py uses for its own post-compress SKETCH routing.
        net.default_task = MERGE
