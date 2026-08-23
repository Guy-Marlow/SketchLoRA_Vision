"""SketchLoRA + growing-rank orthogonal reference (2026-08-21 user design).

Keeps the existing sketch/residual scaffold (models/sketchlora.py: slot 0 =
frozen sketch B_hat@A_hat, slot RESIDUAL = this task's trainable B@A) and the
existing fold-to-rank-10 mechanic at each task boundary -- but the
orthogonality regularizer used WHILE training a new task is not measured
against the sketch itself. It's measured against a SEPARATE, deliberately
smaller, GROWING-rank projection of the sketch (B_orth, A_orth), rebuilt from
scratch at every boundary from the same exact SVD that produces the fold:

  boundary after task t -> B_orth,A_orth = top (orthref_base + orthref_step*t)
  singular components of that boundary's folded composite (clamped to
  lora_rank). Default base=2, step=2: task0's boundary -> rank 2, task1's ->
  rank 4, task2's -> rank 6, task3's -> rank 8, ... i.e. exactly the schedule
  the user specified through task 2's boundary (rank 6), continued by the
  same +2 pattern (confirmed with the user before building).

RATIONALE (user's own framing, confirmed): the rank-truncation probe run
earlier this session showed a rank-10 adapter's classification signal is
concentrated almost entirely in its top 2-3 singular directions -- the rest
is largely redundant. So a new task only needs to avoid the sketch's TOP
few directions to avoid real interference; forcing it away from the WHOLE
rank-10 sketch would be needlessly restrictive (and reintroduces exactly
the "sketch too heavy, no room for new content" problem the "retain"
admission rule was built to fix). The reference grows over time because
more directions become load-bearing as more tasks fold in.

OTHOGONALITY FORMULA -- operates on the DENSE PRODUCTS, not the A-factor
rows (unlike models/sketchlora_align.py's row-wise formula). "Orthogonal"
between two matrices X=B_r@A_r and Y=B_orth@A_orth means their Frobenius
(Hilbert-Schmidt) inner product is zero:
    <X,Y>_F = trace(X^T Y) = trace(A_r^T B_r^T B_orth A_orth)
            = trace((B_r^T B_orth) (A_orth A_r^T))   [cyclic trace identity]
The second form never materializes a [dim,dim] matrix -- B_r^T B_orth is
[r,r_orth] and A_orth A_r^T is [r_orth,r], both tiny -- exact, not an
approximation of the naive dense-matmul formula, just algebraically
reorganized to be cheap. Penalized via abs() and MINIMIZED (added to the
CE loss), same sign convention as every other "orth"-style term in this
project. No "align" variant -- this design only ever pushes away.

FOLD MECHANIC: always exact SVD (torch.linalg.svd, never randomized). The
target fold rank defaults to a FIXED lora_rank (no adaptive/energy_target
concept at all -- this class doesn't expose sketchlora_admission), but can
instead GROW every boundary (2026-08-21 follow-up, sketchlora_orthref_
sketch_fold_base/_step): 1st incorporation -> base, 2nd -> base+step, 3rd
-> base+2*step, etc, uncapped -- see _current_sketch_fold_rank(). Trades
some of the fixed-rank design's memory bound for less interference on
older tasks (the original run showed real, not "virtually zero",
forgetting on the two oldest class blocks). Task 0's "fold" needs no
special case either way: the sketch starts at exactly zero (matching the
base class's own zero-init convention when svd_rank==lora_rank), so
delta_W at task 0 is just the residual's own content (true rank <=
lora_rank already) -- truncating ITS OWN SVD to the target rank recovers
it exactly whenever that target is >= lora_rank (always true for both the
fixed and growing schedules), which is the lossless "skip_compression"
transplant models/sketchlora.py's own _compress() special-cases, falling
out here for free from the general formula. (With a growing schedule whose
first target EXCEEDS the true rank at task 0 specifically -- e.g. base=12
> lora_rank=10 -- a few of the extra kept singular directions are exactly
zero rather than real signal; harmless, just a little wasted width until
later folds have genuine content to fill it.)

DEPLOYED/inference behaviour is completely unchanged from base SketchLoRA
(routes to the SKETCH slot alone, models/sketchlora.py::_deployed_forward,
_eval_adapter) -- B_orth/A_orth are pure training-time scaffolding, never
summed into any forward pass, and are not counted in persistent_state()
(inherited unmodified from models/lora.py -- it only walks the lora_A/B
slot ModuleLists, which B_orth/A_orth are never registered into)."""
import logging
import math
import os

import numpy as np
import torch
from torch import nn, optim
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
        assert self.svd_rank == self.lora_rank, \
            "sketchlora_orthref's INITIAL sketch width must equal lora_rank -- the " \
            "first real _compress() call resizes it to sketch_fold_base regardless, " \
            "so the starting width only matters before that first fold ever runs"
        self.orthref_weight = args.get("sketchlora_orthref_weight", 0.5)
        self.orthref_base = args.get("sketchlora_orthref_base", 2)
        self.orthref_step = args.get("sketchlora_orthref_step", 2)
        # step=0 is allowed -- fixed-rank orth-ref (retention held constant
        # at orthref_base every boundary), same convention as sketch_fold_step
        assert self.orthref_base >= 1 and self.orthref_step >= 0
        # -- penalty formula (2026-08-21 follow-up): "frobenius" (default,
        # original design) treats X=B_r@A_r and Y=B_orth@A_orth as points in
        # R^(dim*dim) and penalizes |<X,Y>_F| -- ONE net inner product, so
        # rows that are aligned in opposite directions can cancel before the
        # abs() ever sees them (the same cancellation risk flagged for
        # "orthogonalize against a SUM of A's" earlier this session).
        # "rowwise" instead mirrors O-LoRA's own formula structure exactly
        # (models/olora.py::_orth_and_l2, |A_t @ A_prev^T|.sum()) but applied
        # to the dense update matrices instead of the low-rank A factors:
        # |X @ Y^T|.sum() -- abs() applied to EVERY pairwise row-dot-product
        # entry before summing, so no cancellation is possible. See
        # _orthref_loss() for the (still cheap) factored computation.
        self.orthref_penalty = args.get("sketchlora_orthref_penalty", "frobenius")
        assert self.orthref_penalty in ("frobenius", "rowwise")
        # -- growing sketch-fold target rank (2026-08-21 follow-up request) --
        # 1st incorporation (boundary after task 0) -> sketch_fold_base (12),
        # 2nd -> +step (14), 3rd -> +2*step (16), etc. Defaults to
        # (lora_rank, 0) -- i.e. UNCHANGED fixed-rank-10 behaviour -- so the
        # original run's config/results stay reproducible; only a config that
        # explicitly sets these two keys gets the growing-rank variant.
        self.sketch_fold_base = args.get("sketchlora_orthref_sketch_fold_base", self.lora_rank)
        self.sketch_fold_step = args.get("sketchlora_orthref_sketch_fold_step", 0)
        assert self.sketch_fold_base >= 1 and self.sketch_fold_step >= 0
        self._orth_ref = None   # list of (B_orth, A_orth) per wrapped module, rebuilt each boundary

    def _current_sketch_fold_rank(self):
        """Target rank the sketch is folded to at the boundary closing task
        self._cur_task. Grows by sketch_fold_step each boundary starting
        from sketch_fold_base -- uncapped (unlike the orth-ref schedule):
        the whole point of this variant is letting the sketch exceed
        lora_rank over time, trading some of the original design's memory
        bound for less interference on older tasks."""
        return self.sketch_fold_base + self.sketch_fold_step * self._cur_task

    def _current_orthref_rank(self):
        """Rank of the reference PRODUCED at the boundary closing task
        self._cur_task -- boundary after task 0 -> orthref_base (2 by
        default), growing by orthref_step each subsequent boundary, clamped
        at THIS SAME boundary's sketch-fold rank (can never ask for more
        components than the fold itself kept -- with the default fixed-
        rank config this clamp is lora_rank, byte-identical to before)."""
        return min(self.orthref_base + self.orthref_step * self._cur_task,
                    self._current_sketch_fold_rank())

    def _orthref_loss(self):
        """Orthogonality penalty between the currently-training residual's
        B@A and the frozen B_orth@A_orth reference (see __init__ and module
        docstring for the algebra of both modes). Gated on self._orth_ref
        being populated -- None until task 0's own boundary runs, matching
        every other orth-style term's `_cur_task > 0` guard in this project
        (task 0 has no history to orthogonalize against)."""
        if self._orth_ref is None:
            return 0.0
        t = self._train_adapter()
        with ce_region("sketchlora_orthref/orthref_loss"):
            reg = 0.0
            idx = 0
            for attn in self._active_attns():
                for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                    A_r = A_list[t].weight   # [r, dim], currently training
                    B_r = B_list[t].weight   # [dim, r]
                    B_orth, A_orth = self._orth_ref[idx]
                    idx += 1
                    if self.orthref_penalty == "frobenius":
                        # <X,Y>_F = trace((B_r^T B_orth)(A_orth A_r^T)), ONE net
                        # scalar -- rows aligned in opposite directions can cancel
                        # before abs() ever sees them.
                        inner = torch.trace((B_r.t() @ B_orth) @ (A_orth @ A_r.t()))
                        reg = reg + torch.abs(inner)
                    else:   # "rowwise" -- O-LoRA's own formula structure, applied
                        # to the dense update matrices: |X @ Y^T|.sum(), abs()'d
                        # PER pairwise row-dot-product entry before summing, so no
                        # cancellation is possible (matches models/olora.py::
                        # _orth_and_l2's |A_t @ A_prev^T|.sum() exactly in form).
                        # Factored to avoid ever needing a [dim,dim] intermediate
                        # until the unavoidable final matmul: X@Y^T = B_r @ (A_r @
                        # A_orth^T) @ B_orth^T.
                        P = A_r @ A_orth.t()        # [r, r_orth]
                        Q = B_r @ P                  # [dim, r_orth]
                        R = Q @ B_orth.t()            # [dim, dim] -- X @ Y^T itself
                        # Normalized by entry count (2026-08-21, user-confirmed):
                        # summing abs() over all dim*dim ~590k entries produces a
                        # raw magnitude ~80x the frobenius mode's single trace (a
                        # smoke test with the unnormalized sum showed real damage
                        # -- task-1 CIL fell to 64.73 vs frobenius's 84.18, same
                        # weight=0.5) -- dividing by entry count restores a
                        # per-entry scale comparable to frobenius mode, isolating
                        # "no cancellation" from "much bigger numbers" so
                        # orthref_weight means roughly the same PRESSURE either way.
                        reg = reg + torch.abs(R).sum() / R.numel()
        return reg

    def _train_core(self, train_loader):
        """models/lora.py::Learner._train's loop, plus the orthref penalty.
        Duplicated (not shared via super()) for the same MRO reason models/
        sketchlora_align.py documents at length: a super()._train() call
        made from WITHIN SketchLoRALearner._train resolves past this
        subclass entirely."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
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

                reg = self._orthref_loss()
                loss = ce + self.orthref_weight * reg

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
                "[SketchLoRA-OrthRef] Task {}, Epoch {}/{} => Loss {:.3f} (CE {:.3f}, orthref {:.4f}), Acc {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs,
                    losses / len(train_loader), ce_run / len(train_loader),
                    reg_run / len(train_loader), train_acc))
        logging.info("[SketchLoRA-OrthRef] Task {} done. Acc {:.2f}, final orthref {:.4f}".format(
            self._cur_task, train_acc, reg_run / len(train_loader)))

    def _compress(self):
        """Fixed-rank exact-SVD fold (always target rank == lora_rank),
        PLUS producing the next boundary's growing-rank orthogonality
        reference from the SAME decomposition (one SVD per module, sliced
        twice -- see module docstring). Replaces models/sketchlora.py's own
        _compress() entirely; none of the base class's admission_rule/
        merge_op branching is used by this class."""
        t = self._train_adapter()
        orth_rank = self._current_orthref_rank()
        new_orth_ref = []
        retained, sigma_next, fro, rhat = [], [], [], []

        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                A_s, B_s = A_list[SKETCH].weight, B_list[SKETCH].weight
                A_r, B_r = A_list[t].weight, B_list[t].weight
                dev, dt = B_s.device, B_s.dtype

                with ce_region("sketchlora_orthref/fold_composite_build"):
                    delta_W = (B_s.float() @ A_s.float()) + (B_r.float() @ A_r.float())

                with ce_region("sketchlora_orthref/fold_exact_svd"):
                    U, S, Vh = torch.linalg.svd(delta_W)

                r = self._current_sketch_fold_rank()
                root_S = S[:r].sqrt()
                B_hat = (U[:, :r] * root_S.unsqueeze(0)).to(dt)
                A_hat = (root_S.unsqueeze(1) * Vh[:r, :]).to(dt)
                if r == B_s.shape[1]:
                    B_s.data.copy_(B_hat.to(dev))
                    A_s.data.copy_(A_hat.to(dev))
                else:
                    # growing-rank variant: the sketch slot's width changed
                    # this boundary -- reallocate, same pattern models/
                    # sketchlora.py's own _compress() uses whenever a fold's
                    # target rank differs from the slot's current width.
                    dim = A_s.shape[1]
                    newA = nn.Linear(dim, r, bias=False).to(dev, dt)
                    newB = nn.Linear(r, dim, bias=False).to(dev, dt)
                    newA.weight.data.copy_(A_hat.to(dev))
                    newB.weight.data.copy_(B_hat.to(dev))
                    for p in list(newA.parameters()) + list(newB.parameters()):
                        p.requires_grad = False
                    A_list[SKETCH] = newA
                    B_list[SKETCH] = newB

                root_S_o = S[:orth_rank].sqrt()
                B_orth = (U[:, :orth_rank] * root_S_o.unsqueeze(0)).detach().clone().to(dev, dt)
                A_orth = (root_S_o.unsqueeze(1) * Vh[:orth_rank, :]).detach().clone().to(dev, dt)
                new_orth_ref.append((B_orth, A_orth))

                if self.sketch_diag:
                    with ce_region("_excluded/sketch_diag"):
                        total_energy = S.pow(2).sum()
                        top_r_energy = S[:r].pow(2).sum()
                        retained.append((top_r_energy / total_energy).item() if total_energy > 0 else 1.0)
                        sigma_next.append(S[r].item() if S.numel() > r else 0.0)
                        fro.append(delta_W.norm().item())
                        rhat.append(r)

                with ce_region("sketchlora_orthref/fold_residual_reset"):
                    nn.init.kaiming_uniform_(A_r, a=math.sqrt(5))
                    nn.init.zeros_(B_r)

        self._orth_ref = new_orth_ref
        self._sketch_populated = True
        if self.sketch_diag:
            self._record_diag(retained, sigma_next, fro, rhat)
        logging.info("[SketchLoRA-OrthRef] Task {} boundary: sketch folded to rank {}, "
                     "new orth-ref rank {}".format(self._cur_task, r, orth_rank))

    def _train(self, train_loader):
        """Copy of models/sketchlora.py::Learner._train's wrapper (task-range
        bookkeeping, block-freezing, compress-at-boundary) with the non-CA
        branch calling _train_core (above) instead of super()._train(), and
        classifier_alignment unsupported (asserted off in scope -- this
        design has no CA-combined variant, matching sketchlora_align.py's
        own scope note)."""
        assert not self.classifier_alignment, \
            "sketchlora_orthref does not support classifier_alignment -- oracle-CIL path only"
        self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
        self._freeze_inactive_blocks()

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
