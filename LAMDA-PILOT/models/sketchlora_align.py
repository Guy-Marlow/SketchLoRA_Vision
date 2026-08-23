"""SketchLoRA + a regularizer relating the newly-trained residual adapter to
the frozen sketch -- a separate codepath in addition to plain SketchLoRA,
not a flag on it (2026-08-20 user request). Two modes, both weight 0.5 by
default, sharing every other line of code:

  "align" (default) -- pulls the residual's A TOWARD the sketch's A. The
    mirror image of O-LoRA's orthogonality penalty (models/olora.py::
    _orth_and_l2), which pulls a new task's A AWAY from every frozen past A.
    "Force the adapters to train in similar directions, minimizing the
    content loss over the backbone."
  "orth" -- pulls the residual's A AWAY from the sketch's A, using O-LoRA's
    OWN formula verbatim (loss = CE + 0.5 * orth, orth = raw un-normalized
    |A_new @ A_sketch^T|.sum()). Requested as a second run to compare
    against "align": does SketchLoRA's own sketch/residual relationship
    behave like O-LoRA's cross-task one if pushed the SAME direction O-LoRA
    pushes, rather than the opposite?

MECHANISM. SketchLoRA keeps two live LoRA structures per wrapped module
(models/sketchlora.py): slot 0 (SKETCH) is the frozen, compressed
accumulated history; slot RESIDUAL (=1 when svd_period=1, the standard
case) is this period's freshly-trained adapter, reset to (kaiming A, zero B)
at every compress boundary and folded into the sketch there. Mode selects
which relationship this file enforces between them every step.

MODE / WEIGHT: config keys sketchlora_align_mode ("align" default, or
"orth") and sketchlora_align_weight (0.5 default), matching O-LoRA's own
lamda_1 convention exactly. Gated on self._cur_task > 0 in BOTH modes,
mirroring O-LoRA's own `if t > 0` guard: at task 0 the sketch is still its
from-construction random Kaiming init (no real history has ever been folded
into it yet), so there is nothing meaningful to align OR orthogonalize
against until after the first compress.

WHY THE TWO MODES USE DIFFERENT FORMULAS, NOT JUST OPPOSITE SIGNS OF THE
SAME EXPRESSION -- read this before changing either. O-LoRA's penalty is
`sum |A_new @ A_prev^T|` (raw, un-normalized r-by-r pairwise dot products),
MINIMIZED -- well-posed because orthogonality has a genuine floor at exactly
0, resisted by the CE loss pulling A_new toward whatever fits the task.
"orth" mode below reuses that exact formula verbatim, since the same
argument applies unchanged (sketch-vs-residual orthogonality also has a
floor at 0). "align" mode CANNOT reuse it as-is: naively negating the same
raw expression to reward high dot products instead of penalizing them is
not well-posed, because raw-dot-product alignment (unlike orthogonality) has
no ceiling -- the optimizer could trivially "win" by growing the residual
A's norm without bound in the sketch's direction, and there is nothing in
this codebase opposing that (every production config sets
sketchlora_lora_wd=0.0, see models/sketchlora.py::_optimizer_param_groups,
so the residual trains with ZERO weight decay). "align" mode therefore
L2-normalizes each row of both A matrices to unit length before the same
r-by-r pairwise structure, making it a genuine "per-direction cosine
similarity, summed over every wrapped module" -- bounded regardless of the
residual's actual parameter norm, capturing DIRECTIONAL alignment without a
magnitude-blowup incentive. If the raw, un-normalized "align" version is
specifically wanted instead, that's a one-line change (drop the two
F.normalize calls in _align_loss), but smoke-test carefully for norm
blow-up first if so.

RETAIN-LINKED WEIGHT (2026-08-20, sketchlora_align_weight_mode="retain_linked"):
when the base class's sketchlora_admission="retain" + sketchlora_retain_anneal
="cosine" are also set, align_weight can be made to track the SAME
instantaneous retention value the retain-mode truncation threshold uses that
task, via linear interpolation over [retain_start,retain_end] ->
[align_weight_min,align_weight_max] (defaults 0.1/0.5) -- see
_current_align_weight() below. Off (align_weight_mode="constant", default) by
default; every existing config is unaffected.

SCOPE: only the standard (classifier_alignment=False) oracle-CIL path is
implemented -- every production/ablation SketchLoRA config in this project
uses that path. CA-combined and streaming (bounded_memory) variants are not
supported; _train falls back to the unmodified inherited behavior for CA
(neither regularizer is applied there) and streaming hooks are left
untouched entirely, matching every oracle-only method added this project
(e.g. models/seqlora_orth.py).
"""

import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from models.sketchlora import Learner as SketchLoRALearner
from models.sketchlora import SKETCH
from utils.ce_profiler import ce_region, run_boundary
from utils.ce2_profiler import ce2_boundary
from utils.toolkit import tensor2numpy


class Learner(SketchLoRALearner):
    def __init__(self, args):
        super().__init__(args)
        self.align_weight = args.get("sketchlora_align_weight", 0.5)
        self.align_mode = args.get("sketchlora_align_mode", "align")
        assert self.align_mode in ("align", "orth"), \
            "sketchlora_align_mode must be 'align' (pull toward sketch, default) " \
            "or 'orth' (push away from sketch, O-LoRA's own formula)"
        # -- orthogonalize B as well as A (2026-08-23 follow-up, "orth" mode
        # only). O-LoRA's own formula (and this file's default "orth" mode,
        # mirroring it) only pushes A_new's ROWS (directions in the INPUT/
        # dim space) away from A_sketch's rows. B is [dim, r] -- its COLUMNS
        # are the corresponding directions in the OUTPUT/dim space -- so the
        # same pairwise-dot-product-abs-sum structure applies transposed:
        # |B_new^T @ B_sketch|.sum(). Off (False) by default -- every
        # existing "orth" config is byte-identical; only a config that
        # explicitly sets this key pays the extra matmul or sees a different
        # loss value.
        self.align_orth_ab = bool(args.get("sketchlora_align_orth_ab", False))
        if self.align_orth_ab and self.align_mode != "orth":
            logging.warning(
                "[SketchLoRA-Align] sketchlora_align_orth_ab=True has no effect "
                "under align_mode=%s -- it only applies to align_mode='orth'.",
                self.align_mode)
        # -- orth-weight linked to the retain-mode retention schedule
        # (2026-08-20 user design): rather than a constant align_weight, the
        # regularizer's strength tracks the base class's OWN instantaneous
        # retention value (_retain_current_value(), the same quantity
        # sketchlora_admission="retain"'s truncation threshold uses) via
        # linear interpolation over [retain_start, retain_end] ->
        # [align_weight_min, align_weight_max]. Rationale (user, verbatim):
        # "as we retain more information, the information we retain should
        # be more valuable (orthogonal and therefore significant); as we
        # retain less information, the information should be less valuable
        # (parallel and therefore likely to coincide over the backbone)" --
        # i.e. this is a function of retention's CURRENT value each task, not
        # an independent schedule over task index -- in a retention-
        # DECREASING run, align_weight decreases right along with it,
        # mirroring retention exactly rather than tracing its own 0.1->0.5
        # ramp regardless of which direction retention is moving.
        self.align_weight_mode = args.get("sketchlora_align_weight_mode", "constant")
        assert self.align_weight_mode in ("constant", "retain_linked")
        self.align_weight_min = args.get("sketchlora_align_weight_min", 0.1)
        self.align_weight_max = args.get("sketchlora_align_weight_max", 0.5)
        if self.align_weight_mode == "retain_linked":
            assert self.retain_anneal == "cosine", \
                "retain_linked reads self._retain_current_value(), which only " \
                "varies under sketchlora_retain_anneal='cosine' -- with retain_anneal " \
                "unset, retention is constant and this mode degenerates to (a slightly " \
                "roundabout way of writing) a constant align_weight"
        if self.classifier_alignment:
            logging.warning(
                "[SketchLoRA-Align] classifier_alignment=True is set, but this "
                "method's regularizer is only wired into the standard (non-CA) "
                "training loop -- CA will run as usual, but the %s term will "
                "NOT be applied for this run.", self.align_mode)

    def _attns(self):
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net.attn_modules()

    def _align_loss(self):
        """"align" mode: mirror of models/olora.py::_orth_and_l2's orth term
        -- the same r-by-r pairwise-dot-product-summed-over-modules
        structure, but rows are L2-normalized first (see module docstring
        for why) and NOT abs()'d, since we want signed alignment (large
        positive), not overlap magnitude."""
        t = self._train_adapter()
        with ce_region("sketchlora_align/align_loss"):
            align = 0.0
            for attn in self._attns():
                for A_list in (attn.lora_A_q, attn.lora_A_v):
                    A_new = A_list[t].weight                    # [r, dim], currently training
                    A_sketch = A_list[SKETCH].weight.detach()    # [r_hat, dim], frozen this task
                    A_new_n = F.normalize(A_new, dim=1)
                    A_sketch_n = F.normalize(A_sketch, dim=1)
                    align = align + (A_new_n @ A_sketch_n.t()).sum()
        return align

    def _orth_loss(self):
        """"orth" mode: models/olora.py::_orth_and_l2's orth term VERBATIM
        (raw, un-normalized, abs()'d r-by-r pairwise dot products, summed
        over every wrapped module) -- applied between the residual and the
        sketch instead of between a new task's adapter and every frozen
        past one.

        When self.align_orth_ab is set, ALSO applies the same structure to
        B (mirrored: B is [dim, r], so its COLUMNS -- not rows -- are the
        directions being compared, hence the transpose on the left operand
        instead of the right: |B_new^T @ B_sketch|.sum(), still one [r,
        r_hat] pairwise matrix per wrapped module). Off by default -- adds
        a second matmul+abs+sum per module only when explicitly requested."""
        t = self._train_adapter()
        with ce_region("sketchlora_align/orth_loss"):
            orth = 0.0
            for attn in self._attns():
                for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                        (attn.lora_A_v, attn.lora_B_v)):
                    A_new = A_list[t].weight                    # [r, dim], currently training
                    A_sketch = A_list[SKETCH].weight.detach()    # [r_hat, dim], frozen this task
                    orth = orth + torch.abs(A_new @ A_sketch.t()).sum()
                    if self.align_orth_ab:
                        B_new = B_list[t].weight                    # [dim, r]
                        B_sketch = B_list[SKETCH].weight.detach()    # [dim, r_hat]
                        orth = orth + torch.abs(B_new.t() @ B_sketch).sum()
        return orth

    def _current_align_weight(self):
        """Constant self.align_weight, or -- when align_weight_mode==
        "retain_linked" -- linearly interpolated from the base class's
        instantaneous retention value (see __init__ for the rationale).
        Linear map is by VALUE, not by schedule direction: retention==
        min(retain_start,retain_end) -> align_weight_min, retention==
        max(retain_start,retain_end) -> align_weight_max, regardless of
        whether retain_start > retain_end (a "down" schedule). BUG FIXED
        2026-08-20, caught by a smoke test before the real runs launched:
        an earlier version used lo,hi = retain_start,retain_end directly,
        which for a down schedule (retain_start=0.9 > retain_end=0.5) put
        hi < lo and INVERTED the mapping -- task 0 of a down run (retention
        =0.9, should map to align_weight_max=0.5) logged align_weight=0.1
        instead. Using min/max of the pair fixes this in both directions."""
        if self.align_weight_mode != "retain_linked":
            return self.align_weight
        retention = self._retain_current_value()
        lo, hi = min(self.retain_start, self.retain_end), max(self.retain_start, self.retain_end)
        frac = 0.0 if hi == lo else (retention - lo) / (hi - lo)
        frac = max(0.0, min(1.0, frac))
        return self.align_weight_min + (self.align_weight_max - self.align_weight_min) * frac

    def _reg_loss(self):
        """Dispatches to the configured mode, gated on self._cur_task > 0
        (see module docstring). Returns a python float 0.0 (not a tensor)
        when the gate is closed -- addable to loss directly either way, a
        plain float 0.0 in a `loss +/- weight * 0.0` expression is a no-op."""
        if self._cur_task <= 0:
            return 0.0
        return self._align_loss() if self.align_mode == "align" else self._orth_loss()

    def _train_core(self, train_loader):
        """models/lora.py::Learner._train's loop, plus the align-to-sketch
        term. Duplicated rather than shared (same reason models/olora.py,
        models/treelora.py, and models/sketchlora.py's own _train_with_ca
        all duplicate this loop instead of hooking the shared base class:
        super()._train() calls made from WITHIN SketchLoRALearner._train
        resolve via type(self).__mro__ starting after SketchLoRALearner,
        landing on models/lora.py directly -- a subclass's own _train
        override is never consulted by that call, so there is no clean
        injection point without either this duplication or a shared-
        infrastructure change to models/lora.py itself, out of scope here)."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        sign = -1.0 if self.align_mode == "align" else 1.0   # align: reward (subtract); orth: penalize (add)
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, ce_run, reg_run, correct, total = 0.0, 0.0, 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                ce = F.cross_entropy(local_logits, local_targets)

                reg = self._reg_loss()
                loss = ce + sign * self._current_align_weight() * reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_run += ce.item()
                reg_run += float(reg)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "[SketchLoRA-Align:{}] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, {} {:.3f}, w={:.3f}), Acc {:.2f}".format(
                    self.align_mode, self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    self.align_mode, reg_run / len(train_loader), self._current_align_weight(), train_acc))
        logging.info("[SketchLoRA-Align:{}] Task {} done. Acc {:.2f}, final {} {:.3f}, align_weight {:.3f}".format(
            self.align_mode, self._cur_task, train_acc, self.align_mode, reg_run / len(train_loader),
            self._current_align_weight()))

    def _train(self, train_loader):
        """Copy of models/sketchlora.py::Learner._train's own wrapper (task-
        range bookkeeping, block-freezing, the R2 pre-boundary probe,
        compress-at-period-boundary, drift extraction) -- see that method
        for the full rationale behind each piece, unchanged here. The one
        substantive difference: the non-CA branch calls self._train_core
        (above, with the align term) instead of super()._train()."""
        self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
        self._freeze_inactive_blocks()

        if self.classifier_alignment:
            self._ca_lazy_init_stats()
            if self.ca_real_mix_frac > 0:
                self._ca_reset_reservoir()
            self._train_with_ca(train_loader)
        else:
            self._train_core(train_loader)

        if getattr(self, "_ce_boundary_ctrl", None) is not None:
            self._ce_pre_boundary_probe(train_loader)

        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        at_last_task = (self._cur_task + 1) >= self._n_run_effective
        if at_period_boundary or at_last_task:
            self._maybe_extract_drift_exact()
            with ce2_boundary(self):
                run_boundary(getattr(self, "_ce_boundary_ctrl", None), "sketchlora_compress",
                            self._compress)
            self._maybe_extract_drift_sketch()
            net = self._network.module if hasattr(self._network, "module") else self._network
            net.default_task = SKETCH
        if self.classifier_alignment:
            with ce2_boundary(self):
                run_boundary(getattr(self, "_ce_boundary_ctrl", None), "sketchlora_ca",
                            self._run_ca_alignment)
