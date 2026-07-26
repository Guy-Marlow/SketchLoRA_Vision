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
from utils.countsketch import countsketch_compress

# fixed-slot convention: 0 = frozen sketch, 1 = trainable residual
SKETCH = 0
RESIDUAL = 1


class Learner(LoRALearner):
    def __init__(self, args):
        super().__init__(args)
        # P = 1.  r̂ (sketch rank) may differ from the residual/adapter rank.
        self.lora_rank = args.get("lora_rank", 10)        # per-task residual rank r
        self.svd_rank = args.get("svd_rank", self.lora_rank)   # sketch target rank r̂ (fixed mode)
        # adaptive-rank mode: if set (e.g. 0.005), each compress keeps the SMALLEST
        # rank retaining (1 - svd_energy_target) of the accumulated delta's energy,
        # resizing the sketch slot.  None => the fixed-rank-r̂ path (untouched).
        self.energy_target = args.get("svd_energy_target", None)
        self.oversampling = args.get("svd_oversampling", 10)
        # optional depth restriction: LoRA only on the first n blocks (else all)
        self.n_lora_blocks = args.get("n_lora_blocks", None)
        # Fixed target-sketch-rank sensitivity (Experiments_Timeline.pdf sec 1.b.ii.3):
        # rather than compressing every task (P=1), train P separate rank-`lora_rank`
        # residuals (slots 1..P) over the frozen sketch before folding all of them in
        # at once -- i.e. R = P * lora_rank is the achieved pre-compression rank for a
        # fixed target R (e.g. R=32, lora_rank=8 -> P=4). Slot count must be
        # provisioned via config ("lora_n_slots": P+1); P=1 (default) is the original,
        # unmodified single-residual behaviour -- every formula below reduces to it
        # exactly when svd_period=1.
        self.svd_period = args.get("svd_period", 1)
        assert self.svd_period >= 1
        # Ablation (Experiments_Timeline.pdf sec 1.b.iii, plan doc §5.3): swap the compression
        # algorithm at each boundary while leaving the surrounding slot/routing machinery (period,
        # residual reset, diagnostics) untouched. "randsvd" (default) is the original, unmodified
        # behaviour above.
        self.merge_op = args.get("merge_op", "randsvd")
        assert self.merge_op in ("randsvd", "exactsvd", "countsketch", "naive_sum", "nocompress")
        # -- Plan A §A5.1/§A5.2 "frozen" SketchLoRA variant (impl_plan_7.25.2026) --
        # All three knobs below default to the ORIGINAL, unmodified behavior --
        # every existing config/run is byte-for-byte unaffected. Set explicitly
        # (sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        # sketchlora_lora_wd=0.0) to opt into the frozen variant Plan C requires.
        # See docs/sketchlora_frozen_variant.md for the full design writeup.
        self.admission_rule = args.get("sketchlora_admission", "global_eps")
        assert self.admission_rule in ("global_eps", "bounded_eviction")
        if self.admission_rule == "bounded_eviction":
            assert self.energy_target is not None, \
                "bounded_eviction is a rank-SELECTION refinement of adaptive (energy_target) " \
                "mode -- it has nothing to bound in fixed-rank mode, where svd_rank already " \
                "pins the rank every merge. Set svd_energy_target to use it."
        self.rank_cap = args.get("sketchlora_rank_cap", None)   # r_max; None = unbounded (old default)
        # Round 2 §2.4: the bounded-eviction FORMULA has two possible readings
        # of the source spec's "t = min(r_residual, k_eps)"; both implemented,
        # switchable, unit-tested (scripts/test_eviction_rule.py). "conformant"
        # (default, matches §2.4's restated spec exactly) reads k_eps as the
        # EVICTION COUNT the energy threshold requests (max(0, composite_rank -
        # keep_rank)); "literal_keeprank" (never used in production, kept only
        # so the rejected reading is a real, runnable code path for the unit
        # test / audit trail) reads k_eps as the KEEP-rank threshold value
        # itself substituted directly as the eviction count -- this fails to
        # evict a meaningful amount precisely when the energy threshold is most
        # aggressive (see the unit test's "responsiveness check").
        self.eviction_reading = args.get("sketchlora_eviction_reading", "conformant")
        assert self.eviction_reading in ("conformant", "literal_keeprank")
        # -- Plan C §C1 lazy_merge (non-default, sign-off gated per §C8 -- runs
        # as a labeled experimental arm only, never the default). Accumulate the
        # residual across multiple bounded-memory cycles instead of folding
        # every cycle; fold only once an internal, boundary-blind saturation
        # statistic trips. See _lazy_should_fold's docstring for the exact
        # statistic and its rationale.
        self.lazy_merge = bool(args.get("lazy_merge", False))
        self.lazy_merge_frac = args.get("lazy_merge_frac", 0.9)
        if self.lazy_merge:
            assert self.energy_target is not None, \
                "lazy_merge's saturation check is an energy-threshold rank measurement -- " \
                "requires svd_energy_target (adaptive mode)."
        # Per-parameter-group weight_decay override for LoRA params only (Plan A
        # §A5.1: "weight_decay = 0 for all LoRA parameters"). None = old behavior
        # (single AdamW group, uniform self.weight_decay for everything, built
        # inline in models/lora.py::_train). See _optimizer_param_groups override.
        self.lora_weight_decay = args.get("sketchlora_lora_wd", None)
        self.cs_rank = args.get("cs_rank", self.svd_rank)
        self.cs_seed = args.get("cs_seed", args["seed"] if not isinstance(args.get("seed"), list)
                                 else args["seed"][0])
        if self.merge_op == "naive_sum":
            assert self.svd_rank == self.lora_rank, \
                "naive_sum keeps rank fixed at the residual rank (no SVD) -- svd_rank must equal " \
                "lora_rank, per Experiments_Timeline.pdf sec 1.b.iii.3"
        if self.merge_op == "nocompress":
            # numerical-rank threshold for "keep everything" (Experiments_Timeline.pdf sec
            # 1.b.iii.4): sigma_i kept iff sigma_i > max(dim) * eps * sigma_1, the standard
            # numerical-rank convention (same one torch/numpy matrix_rank uses internally).
            self.nocompress_eps = args.get("nocompress_eps", 1e-7)
        # last-task detection for _train's compress gate: mirrors trainer.py's own
        # `_n_run = min(stop_after_tasks or nb_tasks, nb_tasks)` exactly, so a forced
        # final compress fires on whichever task index the run loop actually ends on
        # (needed when total tasks isn't a clean multiple of P -- e.g. 50 tasks, P=4
        # leaves a trailing partial period; without this, that tail's residuals never
        # fold and the deployed SKETCH-only TIL eval silently misses them).
        self._n_run_effective = min(args.get("stop_after_tasks") or args["nb_tasks"], args["nb_tasks"])
        # train on sketch(0)+residuals(1..P); both eval paths reduce to the sketch
        self.train_merge = True
        self._network.merge = True
        # if r̂ != r, the sketch slot must hold a rank-r̂ factorisation
        if self.svd_rank != self.lora_rank:
            self._resize_sketch_slot()
        else:
            # _resize_sketch_slot (above) explicitly zeroes BOTH A and B for slot 0; when
            # svd_rank == lora_rank it's skipped and slot 0 keeps the shared scaffold's own
            # per-slot default (vit_lora.py: B zero-init, A KAIMING-init -- true for every
            # slot, sketch included). B=0 makes the PRODUCT B_s@A_s correctly zero regardless
            # of A_s (randsvd/exactsvd, and countsketch's zero-norm-column filter, only ever
            # read that product or are norm-gated), but naive_sum reads A_s directly and would
            # silently sum in this untrained random garbage at task 0 -- zero it explicitly so
            # every merge_op sees the same "no history yet" sketch state.
            for attn in self._all_attns():
                for A_list in (attn.lora_A_q, attn.lora_A_v):
                    nn.init.zeros_(A_list[SKETCH].weight)
        # True once the sketch slot has actually absorbed real content from a
        # merge. False only before the very first compression event, when slot 0
        # is still the zero-initialised placeholder set up above -- see _compress's
        # "nothing to combine yet" branch.
        self._sketch_populated = False
        # -- compression diagnostics (test Corollary 3's structural assumption) --
        # records, per compression event, the singular spectrum of the
        # accumulated delta_W so we can read sigma_{r̂+1} and the retained-energy
        # fraction directly off each truncation.  See Remark 2 condition (iii).
        self.sketch_diag = bool(args.get("sketch_diag", True))
        self._diag_records = []
        seed = args["seed"] if not isinstance(args.get("seed"), list) else args["seed"][0]
        # include the task split (init_cls/increment) so 10-task vs 20-task runs
        # write distinct diagnostic files instead of clobbering each other
        split = "ic{}i{}".format(args.get("init_cls"), args.get("increment"))
        if self.energy_target is not None:
            tag = "adapt{}_b{}_{}".format(self.energy_target, self.n_lora_blocks or "all", split)
        else:
            tag = "r{}_b{}_{}".format(self.svd_rank, self.n_lora_blocks or "all", split)
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

    def _residual_slots(self):
        """Slot indices 1..P (P = svd_period); {RESIDUAL} when P=1 (unchanged)."""
        return list(range(RESIDUAL, RESIDUAL + self.svd_period))

    # Plan A §A5.1: weight_decay=0 for LoRA params specifically (the frozen
    # variant), head keeps the ordinary self.weight_decay. Only takes effect
    # when sketchlora_lora_wd is set in config; otherwise falls back to the
    # base class's single-group behavior (byte-identical to before).
    def _optimizer_param_groups(self):
        if self.lora_weight_decay is None:
            return super()._optimizer_param_groups()
        lora_params, other_params = [], []
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            (lora_params if ("lora_A" in name or "lora_B" in name) else other_params).append(p)
        groups = [{"params": lora_params, "weight_decay": self.lora_weight_decay}]
        if other_params:
            groups.append({"params": other_params, "weight_decay": self.weight_decay})
        return groups

    def _freeze_inactive_blocks(self):
        """When depth-restricted, keep every residual slot frozen on blocks >= n so
        the optimiser never touches them (they stay zero -> no contribution)."""
        if self.n_lora_blocks is None:
            return
        for attn in self._all_attns()[self.n_lora_blocks:]:
            for mlist in (attn.lora_A_q, attn.lora_B_q, attn.lora_A_v, attn.lora_B_v):
                for slot in self._residual_slots():
                    for p in mlist[slot].parameters():
                        p.requires_grad = False

    # -- sample-boundary streaming hooks (reuse _compress) --------------
    # Each chunk trains the residual (slot 1) over the frozen sketch (slot 0); the
    # boundary folds (sketch (+) residual) -> sketch via randomized SVD and resets the
    # residual, exactly like the per-task path but fired on the sample clock so eval
    # always sees the folded sketch. NOT period-aware (always pins slot 1) -- the
    # fixed-target-rank sensitivity sweep (svd_period > 1) is task-boundary-only
    # (Experiments_Timeline.pdf sec 1.b.ii); guarded below rather than silently wrong.
    def _stream_init(self):
        assert self.svd_period == 1, \
            "sketchlora stream/budget mode does not support svd_period > 1 (period is " \
            "defined over TASK boundaries; combine with sample/budget boundaries by " \
            "extending _stream_slot to be period-aware first, or run task-boundary only)"
        self._cur_task = -1                       # diagnostic chunk id used by _compress

    def _stream_slot(self):
        return RESIDUAL

    def _stream_begin_chunk(self, loader):
        self._freeze_inactive_blocks()
        super()._stream_begin_chunk(loader)       # freeze_to_task(1) + fresh optimizer

    def _stream_end_chunk(self, loader):
        self._cur_task += 1
        self._freeze_inactive_blocks()
        if self.lazy_merge and not self._lazy_should_fold():
            # Accumulate: DON'T compress, don't reset the residual -- leave its
            # (now further-trained) weights exactly as they are. The next
            # cycle's _stream_begin_chunk only re-establishes trainability +
            # a fresh optimizer (StreamMixin's default), it does not touch the
            # residual's weights, so training simply continues on the SAME
            # residual across cycles until this returns True.
            return
        self._compress()                          # fold -> sketch, reset residual (folded)

    @torch.no_grad()
    def _lazy_should_fold(self):
        """Plan C §C1 lazy_merge saturation check: an internal, boundary-blind
        statistic (no data volume, cycle count, or real-task information read)
        -- fold once the residual's OWN occupied rank, measured by the SAME
        energy-threshold rule used for the main compression but applied to the
        residual's factor product (B_r @ A_r) in isolation, reaches
        `lazy_merge_frac` of its allocated rank budget (self.lora_rank). Once a
        rank-`lora_rank` residual is using most of its own budget to represent
        the data trained into it so far, it has little room left to absorb
        materially new signal without truncation -- an intentionally simple,
        self-contained proxy for "residual energy/rank saturates" (Plan C's own
        phrasing), not a claim of optimality."""
        ratios = []
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                for slot in self._residual_slots():
                    A_r, B_r = A_list[slot].weight, B_list[slot].weight
                    delta = (B_r @ A_r).float()
                    S = torch.linalg.svdvals(delta)
                    total = S.pow(2).sum()
                    if total <= 0:
                        continue
                    cum = torch.cumsum(S.pow(2), 0) / total
                    r_hat = int((cum < (1.0 - self.energy_target)).sum().item()) + 1
                    ratios.append(r_hat / self.lora_rank)
        if not ratios:
            return False
        return (sum(ratios) / len(ratios)) >= self.lazy_merge_frac

    # -- adapter routing (override the lora.Learner indirection) --------
    def _train_adapter(self):
        # slot 1 on task 0, slot 2 on task 1, ..., slot P on task P-1, then back to
        # slot 1 -- merge=True sums slots 0..this inclusive (models/lora.py's
        # _lora_delta), which is exactly "sketch + every residual filled so far this
        # period" since slots are visited in strictly increasing order within a period
        # and all reset to zero at the period's compress event (see _compress).
        return RESIDUAL + (self._cur_task % self.svd_period)

    def _eval_adapter(self, task):
        return SKETCH            # TIL routes to the single compressed sketch

    def _deployed_forward(self, inputs):
        """CIL/FLOPs-measurement forward (utils/metrics_logger.py's record_inference_cost,
        called once after the final task -- by then _compress() has already run for that
        task too). Unlike the shared models/lora.py version, this does NOT call
        self._train_adapter() (which always points at the residual slot, since that's the
        correct routing DURING training) -- deployed/evaluated inference happens
        post-compress, when the residual is a reset (kaiming A, zero B) no-op, so routing
        through the sketch alone is bit-exact and matches the paper's O(r̂d) inference cost
        instead of training's O((r̂+r)d)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        return net(inputs, task=SKETCH, merge=True)

    # -- train then compress, but ONLY at a period boundary (or the run's last task)
    # -- eval runs before after_task) --------------
    def _train(self, train_loader):
        self._freeze_inactive_blocks()   # re-freeze before the optimiser is built
        super()._train(train_loader)
        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        at_last_task = (self._cur_task + 1) >= self._n_run_effective
        if at_period_boundary or at_last_task:
            self._compress()
            # Eval (CIL's bare net(inputs), routed via LoRAVitNet.default_task -- see
            # utils/inc_net.py's _resolve) runs after this returns, before after_task().
            # _compress() just reset every residual slot to (kaiming A, zero B) -- a
            # mathematically exact no-op contribution -- so summing sketch+residual at
            # eval time is provably identical to the sketch alone, just twice the
            # matmuls. Route default_task to the sketch slot so eval pays the paper's
            # O(r̂d) inference cost (Table 1) instead of training's O((r̂+r)d); this
            # cannot change any computed value (adding a zero-valued branch is a no-op),
            # only skips computing it. Reset again next task by incremental_train's own
            # `default_task = self._train_adapter()` (models/lora.py) before any training
            # happens, so this never leaks into the next task's training routing.
            net = self._network.module if hasattr(self._network, "module") else self._network
            net.default_task = SKETCH

    @torch.no_grad()
    def _compress(self):
        """Compress (sketch ⊕ residuals 1..P) -> sketch, per layer & proj. The compression
        ALGORITHM is selected by ``self.merge_op`` (Experiments_Timeline.pdf sec 1.b.iii, plan
        doc §5.3); everything else below (period/residual-reset/diagnostics machinery) is shared
        across every merge_op:

        - ``randsvd`` (default): randomized SVD via ``utils.randsvd.rand_svd``.
        - ``exactsvd``: full ``torch.linalg.svd`` + truncate (same rank-selection logic as
          randsvd, exact instead of approximate reconstruction).
        - ``countsketch``: ``utils.countsketch.countsketch_compress`` -- hashes the concatenated
          factors' rank dimension down to ``cs_rank`` buckets; does NOT form an optimal
          low-rank projection (a random signed merge of components instead).
        - ``naive_sum``: no SVD at all, literal running sum of the raw B/A factor matrices
          (requires ``svd_rank == lora_rank``) -- does not preserve delta_W, deliberately.
        - ``nocompress``: keep every singular direction above the numerical-rank threshold
          (grows the sketch's rank every boundary; the sketch slot is variable-width like
          adaptive mode).

        Rank selection: fixed mode (``energy_target`` unset) truncates to ``svd_rank`` (the
        fixed-rank sensitivity sweep sets this to the FULL target rank R; with P residuals of
        rank ``lora_rank`` each, the accumulated pre-compression rank is P*lora_rank -- e.g.
        R=32, lora_rank=8 -> P=4 -- per Experiments_Timeline.pdf sec 1.b.ii.3). Adaptive mode
        (``svd_energy_target`` set, P always 1 -- doesn't combine with the fixed-period sweep):
        keep the smallest rank retaining (1 - ε) of the energy, resizing the sketch slot
        (variable-width) so memory tracks the intrinsic rank. ``nocompress``/``naive_sum``
        override rank selection with their own rule regardless of ``energy_target``.

        Diagnostics (``retained``/``sigma_next``/``fro``/``rhat``) are always measured from the
        ACTUAL reconstruction (B_hat @ A_hat vs. the true delta_W), not assumed from the
        idealized truncated-SVD value -- naive_sum/countsketch don't achieve the optimal
        projection at their nominal rank, so this must be measured to stay comparable across
        merge_op ablations."""
        retained, sigma_next, fro, rhat = [], [], [], []   # per (layer,proj) diagnostics
        residual_slots = self._residual_slots()
        # Nothing to combine yet: before the sketch has ever been populated, a
        # single residual slot is the ONLY contributor to delta_W (the sketch
        # slot is still the zero placeholder) -- there is no history to fold in,
        # so running any merge algorithm (SVD truncation, count-sketch hashing,
        # ...) on it can only discard information for zero compression benefit.
        # Real sketching starts once there are >=2 things to combine: task 1's
        # (sketch + residual), or (with svd_period=P>1) the first boundary's P
        # residuals even though the sketch itself is still zero at that point.
        skip_compression = (not self._sketch_populated) and len(residual_slots) == 1
        full_svd_needed = (not skip_compression) and self.merge_op in ("exactsvd", "nocompress")
        need_svdvals = (not skip_compression) and (not full_svd_needed) and \
            (self.sketch_diag or self.energy_target is not None)
        module_idx = 0     # unique per (layer,proj) -- seeds countsketch's hash/sign draw
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_s, B_s = A_list[SKETCH].weight, B_list[SKETCH].weight     # [r,d],[d,r]
                dev, dt = B_s.device, B_s.dtype
                delta_W = B_s @ A_s                                        # [d, d], unscaled
                for slot in residual_slots:
                    A_r, B_r = A_list[slot].weight, B_list[slot].weight
                    delta_W = delta_W + B_r @ A_r

                if skip_compression:
                    # transplant the lone residual into slot 0 verbatim, at its
                    # own native rank -- no SVD, no loss.
                    only_slot = residual_slots[0]
                    B_hat = B_list[only_slot].weight.clone()
                    A_hat = A_list[only_slot].weight.clone()
                    final_rank = B_hat.shape[1]
                    if self.sketch_diag:
                        fro_delta = delta_W.float().norm()
                        retained.append(1.0)
                        sigma_next.append(0.0)
                        fro.append(fro_delta.item())
                        rhat.append(final_rank)
                    B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                    if final_rank == B_s.shape[1]:
                        B_s.data.copy_(B_hat)
                        A_s.data.copy_(A_hat)
                    else:
                        newA = nn.Linear(delta_W.shape[0], final_rank, bias=False).to(dev, dt)
                        newB = nn.Linear(final_rank, delta_W.shape[0], bias=False).to(dev, dt)
                        newA.weight.data.copy_(A_hat)
                        newB.weight.data.copy_(B_hat)
                        for p in list(newA.parameters()) + list(newB.parameters()):
                            p.requires_grad = False
                        A_list[SKETCH] = newA
                        B_list[SKETCH] = newB
                    for slot in residual_slots:
                        nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                        nn.init.zeros_(B_list[slot].weight)
                    module_idx += 1
                    continue

                S = None
                if full_svd_needed:
                    U, S, Vh = torch.linalg.svd(delta_W.float())          # reused for diag + recon
                elif need_svdvals:
                    S = torch.linalg.svdvals(delta_W.float())             # full spectrum, desc

                # -- choose the target rank for this compression --
                if self.merge_op == "naive_sum":
                    r_hat_t = self.svd_rank                                # == lora_rank (asserted)
                elif self.merge_op == "nocompress":
                    # numerical rank of delta_W (Experiments_Timeline.pdf sec 1.b.iii.4): keep
                    # EVERY singular direction above the standard numerical-rank threshold --
                    # this is "no compression" in the sense that nothing informative is dropped,
                    # not literally infinite rank.
                    thresh = max(delta_W.shape) * self.nocompress_eps * (S[0].item() if S.numel() else 0.0)
                    r_hat_t = max(1, int((S > thresh).sum().item()))
                elif self.energy_target is not None:
                    total = S.pow(2).sum()
                    if total > 0:
                        cum = torch.cumsum(S.pow(2), 0) / total
                        k_eps = int((cum < (1.0 - self.energy_target)).sum().item()) + 1
                    else:
                        k_eps = 1
                    k_eps = max(1, min(k_eps, delta_W.shape[0]))
                    if self.admission_rule == "bounded_eviction":
                        # Plan A §A5.2 / Round 2 §2.4: never evict more than the
                        # residual's OWN just-added rank per merge (bounds eviction to
                        # <= what was just contributed, so rank is monotone non-
                        # decreasing below the cap -- the pure-global-eps branch below
                        # can otherwise evict far more than that in one merge if the
                        # composite's post-fold energy spectrum happens to concentrate
                        # differently, which is the "retroactive mass-eviction" /
                        # post-peak rank collapse A5.2 exists to fix).
                        #
                        # RESOLVED (Round 2 §2.4, restates the spec-conformant rule
                        # explicitly): below cap, evict t = min(r_residual, k_eps)
                        # trailing directions; at cap, evict exactly (composite_rank -
                        # r_max). Two readings of k_eps exist and both are implemented,
                        # switchable via sketchlora_eviction_reading (default
                        # "conformant"), unit-tested in scripts/test_eviction_rule.py:
                        #   conformant: k_eps = requested EVICTION count (naive_evict
                        #     below, = max(0, composite_rank - keep_rank)) -- rank
                        #     tracks the energy signal responsively, selected as correct.
                        #   literal_keeprank: k_eps = the KEEP-rank threshold itself,
                        #     substituted directly as the eviction count -- evicts almost
                        #     nothing whenever the threshold is aggressive (small
                        #     keep-rank), the opposite of the rule's purpose; kept only
                        #     as a documented, tested, never-used-in-production path.
                        prev_rank = A_s.shape[0]
                        residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                        composite_rank = prev_rank + residual_total
                        cap = self.rank_cap if self.rank_cap is not None else composite_rank
                        if composite_rank > cap:
                            # at/above the cap: evict exactly enough to return to r_max,
                            # even if that's more than residual_total.
                            evict = composite_rank - cap
                        elif self.eviction_reading == "literal_keeprank":
                            evict = min(residual_total, k_eps)
                        else:
                            naive_evict = max(0, composite_rank - k_eps)
                            evict = min(residual_total, naive_evict)
                        r_hat_t = max(1, composite_rank - evict)
                    else:
                        r_hat_t = k_eps
                        if self.rank_cap is not None:
                            # A5.1's hard cap lands independently of the admission-rule
                            # sign-off (A5.2) -- applies as a plain clamp here too.
                            r_hat_t = min(r_hat_t, self.rank_cap)
                elif self.merge_op == "countsketch":
                    r_hat_t = self.cs_rank
                else:
                    r_hat_t = self.svd_rank

                # -- compute the merged (B_hat, A_hat) factor pair per the chosen algorithm --
                if self.merge_op == "naive_sum":
                    # no SVD at all: literal running sum of the raw factor matrices
                    # (Experiments_Timeline.pdf sec 1.b.iii.3) -- does NOT preserve
                    # delta_W = B_hat @ A_hat in general; that is the deliberate point of this
                    # ablation, not a bug.
                    B_hat, A_hat = B_s.clone(), A_s.clone()
                    for slot in residual_slots:
                        B_hat = B_hat + B_list[slot].weight
                        A_hat = A_hat + A_list[slot].weight
                elif self.merge_op == "countsketch":
                    B_ws = [B_s] + [B_list[slot].weight for slot in residual_slots]
                    A_ws = [A_s] + [A_list[slot].weight for slot in residual_slots]
                    seed = (int(self.cs_seed) * 1_000_003 + (self._cur_task + 1) * 9176
                            + module_idx) % (2 ** 63 - 1)
                    B_hat, A_hat = countsketch_compress(B_ws, A_ws, r_hat_t, seed)
                elif full_svd_needed:
                    root_S = S[:r_hat_t].sqrt()
                    B_hat = U[:, :r_hat_t].to(dt) * root_S.to(dt).unsqueeze(0)
                    A_hat = root_S.to(dt).unsqueeze(1) * Vh[:r_hat_t, :].to(dt)
                else:
                    B_hat, A_hat = rand_svd(delta_W, r_hat_t, self.oversampling)
                B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                final_rank = B_hat.shape[1]

                if self.sketch_diag:
                    # ACTUAL achieved retained energy from the real reconstruction, not the
                    # idealized truncated-SVD value -- naive_sum/countsketch do not achieve the
                    # optimal projection at their nominal rank, so this must be measured, not
                    # assumed, to make diagnostics comparable across merge_op ablations.
                    recon_err = (delta_W.float() - (B_hat.float() @ A_hat.float())).norm()
                    fro_delta = delta_W.float().norm()
                    retained.append(1.0 - (recon_err / fro_delta).item() ** 2 if fro_delta > 0 else 1.0)
                    sigma_next.append(S[final_rank].item() if (S is not None and S.numel() > final_rank) else 0.0)
                    fro.append(fro_delta.item())
                    rhat.append(final_rank)

                if final_rank == B_s.shape[1]:
                    # rank unchanged -> in-place copy (fixed mode, or no growth)
                    B_s.data.copy_(B_hat)
                    A_s.data.copy_(A_hat)
                else:
                    # rank changed (adaptive mode, or nocompress's growing sketch) -> replace
                    # slot-0 Linears (variable width)
                    dim = delta_W.shape[0]
                    newA = nn.Linear(dim, final_rank, bias=False).to(dev, dt)
                    newB = nn.Linear(final_rank, dim, bias=False).to(dev, dt)
                    newA.weight.data.copy_(A_hat)
                    newB.weight.data.copy_(B_hat)
                    for p in list(newA.parameters()) + list(newB.parameters()):
                        p.requires_grad = False
                    A_list[SKETCH] = newA
                    B_list[SKETCH] = newB
                # reset every residual slot: kaiming A, zero B -> clean + eval no-op.
                # All P must reset (not just the one that just trained) since the NEXT
                # period's _train_adapter() revisits slot 1 first and merge=True would
                # otherwise re-sum a stale prior-period residual left non-zero.
                for slot in residual_slots:
                    nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                    nn.init.zeros_(B_list[slot].weight)
                module_idx += 1
        self._sketch_populated = True
        if self.sketch_diag:
            self._record_diag(retained, sigma_next, fro, rhat)

    def _record_diag(self, retained, sigma_next, fro, rhat):
        """Aggregate + persist the per-compression singular-spectrum stats."""
        import numpy as np
        rec = {
            "task": self._cur_task,
            "retained_energy": retained,        # frac of ||ΔW||² kept by top-r̂
            "sigma_next": sigma_next,           # σ_{r̂+1}(ΔW), per (layer,proj)
            "fro": fro,                         # ||ΔW||_F, per (layer,proj)
            "r_hat": rhat,                      # rank chosen this compress, per (layer,proj)
            "retained_mean": float(np.mean(retained)),
            "retained_min": float(np.min(retained)),
            "fro_mean": float(np.mean(fro)),
            "r_hat_mean": float(np.mean(rhat)) if rhat else None,
            "r_hat_max": int(np.max(rhat)) if rhat else None,
            "r_hat_total": int(np.sum(rhat)) if rhat else None,
        }
        self._diag_records.append(rec)
        os.makedirs(os.path.dirname(self._diag_path), exist_ok=True)
        with open(self._diag_path, "w") as f:
            json.dump(self._diag_records, f, indent=2)
        rh = (" | r_hat mean={:.1f} total={}".format(rec["r_hat_mean"], rec["r_hat_total"])
              if rec["r_hat_mean"] is not None else "")
        logging.info(
            "[SketchDiag] task {}: retained-energy mean={:.3f} min={:.3f} | "
            "||ΔW||_F mean={:.3f}{}".format(
                self._cur_task, rec["retained_mean"], rec["retained_min"], rec["fro_mean"], rh))
