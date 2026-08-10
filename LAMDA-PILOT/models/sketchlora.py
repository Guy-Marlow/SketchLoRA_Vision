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

SketchLoRA v2 (impl_plan_7.28.2026): decisions locked by the bolt-on ablation
data, recorded here per plan sec 0 --
  * FD shrinkage (fd_shrinkage=True) is OFF the v2 path: cost 2-4 top1 at 15T
    locally, no accuracy benefit found. Retained as the rank/accuracy
    compression dial, not refuted outright -- e.g. lazy-merge@100MB: 65.82
    top1 at r_hat~22 vs 66.42 at r_hat~93 baseline is the tradeoff figure to
    report. Its long-horizon (>15-task) claim remains untested, so it is
    documented as off-path, not disproven.
  * An annealing-epsilon admission proposal was REJECTED: it requires a
    known clock/horizon to anneal against, which reintroduces exactly the
    stream-length-oracle dependency this project criticizes InfLoRA's
    DualGPM for -- it treats the symptom (rank growth) rather than the cause
    (which directions get evicted).
  * A tree-merge (pairwise balanced merging) proposal is DEFERRED to v3+.
    Structurally this is the mergeable-summaries / distributed-FD construction
    (Agarwal et al. 2012; Ghashami et al. SICOMP 2016's merge theorem): error
    composes additively over a log-depth root path, so retained state becomes
    O(log K) rather than O(1), and loss is scheduled fairly across the merge
    tree rather than eliminated. Worth a pilot only if research-time reserve
    remains after the v2 money runs.
  * The two 2026-07-28 admission-rule ablations (guaranteed_admission,
    force_increase) are RETIRED and replaced by admission_rule="floor" (sec 1
    below) -- a single codepath that fixes force_increase's at-cap floor-
    collapse bug by construction instead of patching the eviction-count
    formula a second time.
"""

import json
import logging
import math
import os
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.lora import Learner as LoRALearner, num_workers
from utils.toolkit import tensor2numpy

# trusted randomized-SVD implementation (vendored into utils/ for self-containment)
from utils.randsvd import rand_svd, rand_svd_probe, factors_from_probe
from utils.countsketch import countsketch_compress
from utils.admission import floor_admission_merge
# *** UNTESTED as of 2026-08-03 *** -- measured-CE region tagging
# (docs/ce_profiling_implementation_plan.md sec 4.1, sec 5 Step 2). ce_region()
# is a no-op unless a profiling session is active (utils/ce_profiler.py), so
# these tags are safe to leave in permanently and do not change any computed
# value -- only what utils/ops_ledger.py's measured_step_regions/
# measured_boundary_regions record.
from utils.ce_profiler import ce_region, run_boundary, run_step_narrow
from utils.ce2_profiler import ce2_boundary

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
        assert self.merge_op in ("randsvd", "exactsvd", "countsketch", "naive_sum",
                                  "nocompress", "reduce_merge")
        # -- FD shrinkage bolt-on (impl_plan_7.27.2026 sec 1.1). Only meaningful for
        # SVD-truncation merge_ops -- "Sigma[l]" (the first discarded singular value)
        # has no analogue for naive_sum (no SVD) or countsketch (hashed, not truncated),
        # so it's a documented no-op there rather than a hard error.
        self.fd_shrinkage = bool(args.get("fd_shrinkage", False))
        if self.fd_shrinkage and self.merge_op not in ("randsvd", "exactsvd"):
            logging.warning(
                "[SketchLoRA] fd_shrinkage=True but merge_op=%s has no truncated-SVD "
                "spectrum to shrink -- fd_shrinkage will be a no-op for this run.",
                self.merge_op)
        self._fd_cumulative_rent = []   # per-module running total of FD "rent" charged
        # -- Plan A §A5.1/§A5.2 "frozen" SketchLoRA variant (impl_plan_7.25.2026) --
        # All three knobs below default to the ORIGINAL, unmodified behavior --
        # every existing config/run is byte-for-byte unaffected. Set explicitly
        # (sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        # sketchlora_lora_wd=0.0) to opt into the frozen variant Plan C requires.
        # See docs/sketchlora_frozen_variant.md for the full design writeup.
        self.admission_rule = args.get("sketchlora_admission", "global_eps")
        assert self.admission_rule in ("global_eps", "bounded_eviction", "floor")
        if self.admission_rule in ("bounded_eviction", "floor"):
            assert self.energy_target is not None, \
                "bounded_eviction/floor are rank-SELECTION refinements of adaptive " \
                "(energy_target) mode -- nothing to bound in fixed-rank mode, where svd_rank " \
                "already pins the rank every merge. Set svd_energy_target to use it."
        # reduce_merge (2026-08-05, "reduce then merge" ablation -- see _compress)
        # never evicts anything from the existing sketch (only ever adds to it,
        # then re-expresses the sum losslessly), so admission_rule's whole
        # question -- "how much of the EXISTING sketch to evict this merge" --
        # doesn't apply to it. Same warning pattern as fd_shrinkage's merge_op
        # compatibility check above.
        if self.merge_op == "reduce_merge" and self.admission_rule != "global_eps":
            logging.warning(
                "[SketchLoRA] merge_op=reduce_merge ignores sketchlora_admission=%s -- "
                "this merge_op never evicts from the existing sketch, so the admission "
                "rule has no effect for this run.", self.admission_rule)
        # -- admission rule v2: floor + cap-turnover (impl_plan_7.28.2026 sec 1).
        # Single codepath REPLACING the two 2026-07-28-direction ablations
        # guaranteed_admission (reserved-slot protection, no floor-survives-cap
        # guarantee) and force_increase (floor via an eviction-count adjustment
        # that a live H200-bound run proved does NOT survive the at-cap branch --
        # evict = composite_rank - cap there ignored the floor entirely, so
        # force_increase silently degenerated to plain bounded_eviction once rank
        # hit rank_cap). "floor" fixes this by construction: protect the top-k
        # directions of the residual's component orthogonal to the pre-merge
        # sketch (guaranteed_admission's idea), but size the energy-fill budget
        # as r_hat_t - k (r_hat_t from the UNCHANGED bounded_eviction formula)
        # so the protected k are additive to, never competing with, the target
        # rank -- they cannot be evicted at any cap. See utils/admission.py for
        # the full algorithm + rationale; unit tests:
        # scripts/test_floor_admission_synthetic.py (floor survives at cap),
        # scripts/test_floor_golden_bitexact.py (k=0 == bounded_eviction, bit-exact).
        self.admission_floor_k = args.get("sketchlora_admission_floor_k", 1)
        if self.admission_rule == "floor":
            assert self.energy_target is not None, \
                "floor fills its non-protected slots by the same energy-threshold " \
                "rule as adaptive mode -- requires svd_energy_target"
            assert self.merge_op == "randsvd", \
                "floor is only implemented for merge_op='randsvd' (our project " \
                "default) -- exactsvd/countsketch/naive_sum/nocompress are not " \
                "wired into utils.admission.floor_admission_merge"
            assert self.admission_floor_k >= 1
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
        # -- lazy merge (impl_plan_7.27.2026 sec 1.2). Generalizes/supersedes the
        # earlier Plan C §C1 boolean+frac design (kept, unrenamed in behavior, as
        # the "legacy_saturation" mode below, for exact backward compat with any
        # config still passing a bare bool -- "lazy_merge": true means exactly what
        # it always meant). New surface: lazy_merge in {"off","period","plateau"}.
        #   period:  ALIASES svd_period (does not duplicate it) -- lazy_merge_period
        #            just sets self.svd_period, and streaming folds at period
        #            boundaries exactly like the oracle path's at_period_boundary
        #            check (see _stream_slot/_stream_end_chunk).
        #   plateau: NEW. utils.lazy.PlateauTracker -- folds once the residual's
        #            own drift plateaus for 2 consecutive cycles, or
        #            lazy_merge_max_holdoff cycles have passed since the last fold.
        raw_lazy = args.get("lazy_merge", "off")
        if isinstance(raw_lazy, bool):
            self.lazy_merge_mode = "legacy_saturation" if raw_lazy else "off"
        else:
            assert raw_lazy in ("off", "period", "plateau"), \
                "lazy_merge must be a bool (legacy) or one of off/period/plateau"
            self.lazy_merge_mode = raw_lazy
        self.lazy_merge = self.lazy_merge_mode != "off"   # legacy boolean, kept for gating below
        self.lazy_merge_frac = args.get("lazy_merge_frac", 0.9)             # legacy_saturation only
        self.lazy_merge_period = args.get("lazy_merge_period", None)        # period only
        self.lazy_merge_delta = args.get("lazy_merge_delta", 0.05)          # plateau only
        self.lazy_merge_max_holdoff = args.get("lazy_merge_max_holdoff", 10)  # period-mode: unused; plateau: staleness cap
        if self.lazy_merge_mode == "legacy_saturation":
            assert self.energy_target is not None, \
                "lazy_merge's saturation check is an energy-threshold rank measurement -- " \
                "requires svd_energy_target (adaptive mode)."
        if self.lazy_merge_mode == "period":
            assert self.lazy_merge_period is not None and self.lazy_merge_period >= 1, \
                "lazy_merge=period requires lazy_merge_period >= 1"
            configured_svd_period = args.get("svd_period", 1)
            if configured_svd_period != 1 and configured_svd_period != self.lazy_merge_period:
                raise ValueError(
                    "svd_period and lazy_merge_period are the SAME underlying knob "
                    "(impl_plan_7.27.2026 sec 1.2: 'this generalizes the existing period "
                    "hyperparameter; alias them, do not duplicate') -- set only one, or "
                    "set both to the same value")
            self.svd_period = self.lazy_merge_period   # overrides the svd_period=1 default above
        self._plateau_tracker = None
        if self.lazy_merge_mode == "plateau":
            assert self.svd_period == 1, \
                "plateau mode is a single-residual accumulate-until-triggered design " \
                "(impl_plan_7.27.2026 sec 1.2) -- does not combine with svd_period>1"
            from utils.lazy import PlateauTracker
            self._plateau_tracker = PlateauTracker(self.lazy_merge_delta, self.lazy_merge_max_holdoff)
        # -- classifier alignment (impl_plan_7.27.2026 sec 1.3). SLCA-style,
        # exemplar-free: online per-class {mean, diagvar} over penultimate
        # features (utils.ca.ClassStats), head-only realignment on pseudo-
        # features sampled from those per-class Gaussians, run every stream
        # cycle (see _stream_end_chunk). ClassStats needs self._network.fc's
        # width, which does not exist yet at __init__ time (update_fc runs
        # later, in bounded_memory_run) -- deferred to _stream_init.
        self.classifier_alignment = bool(args.get("classifier_alignment", False))
        self.ca_steps = args.get("ca_steps", 300)
        self.ca_batch = args.get("ca_batch", 128)
        self.ca_lr = args.get("ca_lr", 1e-3)
        self.ca_store = args.get("ca_store", "mean_diagvar")
        if self.classifier_alignment:
            assert self.ca_store == "mean_diagvar", \
                "only ca_store='mean_diagvar' is implemented (impl_plan_7.27.2026 sec 1.3 default)"
        # -- CA repair sweep, v2 variants (impl_plan_7.28.2026 sec 2) --
        # (b) covariance mode: "diag" (v1, unchanged default) / "shared_full" /
        # "low_rank_diag" -- see utils/ca.py::ClassStats.
        self.ca_cov_mode = args.get("ca_cov_mode", "diag")
        # (a) ca_steps sweep is just ca_steps itself (already a config knob);
        # early stopping against a held-out pseudo-feature batch, checked
        # every 10 steps, patience in units of checks (None = off, v1 behavior).
        self.ca_early_stop_patience = args.get("ca_early_stop_patience", None)
        self.ca_val_batch = args.get("ca_val_batch", None)   # None -> defaults to ca_batch
        # (c) real-feature mixing: fraction of each align_head batch drawn from
        # a bounded per-cycle reservoir of ACTUAL current-cycle features
        # (collected in _bounded_train_epoch) instead of sampled pseudo-features.
        self.ca_real_mix_frac = args.get("ca_real_mix_frac", 0.0)
        self.ca_real_reservoir_size = args.get("ca_real_reservoir_size", 512)
        # (d) logit-adjustment-ONLY arm: no head retraining at all -- additive
        # per-class prior correction from stored class COUNTS (2026-07-28
        # user-resolved reading of the plan's "counts/means" phrasing; see
        # utils/ca.py::logit_adjust_bias). Mutually exclusive with the
        # ordinary align_head path (this is a structurally different arm, not
        # a combinable knob).
        self.ca_logit_adjust_only = bool(args.get("ca_logit_adjust_only", False))
        self.ca_logit_adjust_tau = args.get("ca_logit_adjust_tau", 1.0)
        self._ca_logit_correction = None   # running applied bias, for delta-tracking
        self._ca_real_buffer_feats = None  # reservoir tensors, built lazily
        self._ca_real_buffer_labels = None
        self._ca_real_buffer_n_seen = 0
        self._ca_stats = None
        # Per-parameter-group weight_decay override for LoRA params only (Plan A
        # §A5.1: "weight_decay = 0 for all LoRA parameters"). None = old behavior
        # (single AdamW group, uniform self.weight_decay for everything, built
        # inline in models/lora.py::_train). See _optimizer_param_groups override.
        self.lora_weight_decay = args.get("sketchlora_lora_wd", None)
        # -- embedding-drift extraction (t-SNE study, 2026-08-06) -- opt-in, off
        # by default. When set, at every compress boundary we dump (a) the
        # current task's OWN test-set embeddings as produced by the just-trained,
        # not-yet-compressed adapter (sketch_{n-1} frozen + residual_n, exactly
        # the state training used) and (b) the sketch's embeddings, right after
        # compression, on every task seen so far (task r <= n) -- isolating
        # compression-induced drift from later-task drift. Each group is written
        # to its own .npz file as soon as it's computed (not batched at the end)
        # so t-SNE tuning can start before training finishes.
        self.embed_drift_dir = args.get("embed_drift_dir", None)
        if self.embed_drift_dir is not None:
            os.makedirs(self.embed_drift_dir, exist_ok=True)
        self._task_class_ranges = {}   # task idx -> (known, total) class bounds
        self.cs_rank = args.get("cs_rank", self.svd_rank)
        self.cs_seed = args.get("cs_seed", args["seed"] if not isinstance(args.get("seed"), list)
                                 else args["seed"][0])
        if self.merge_op == "naive_sum":
            assert self.svd_rank == self.lora_rank, \
                "naive_sum keeps rank fixed at the residual rank (no SVD) -- svd_rank must equal " \
                "lora_rank, per Experiments_Timeline.pdf sec 1.b.iii.3"
        if self.merge_op in ("nocompress", "reduce_merge"):
            # numerical-rank threshold for "keep everything" (Experiments_Timeline.pdf sec
            # 1.b.iii.4): sigma_i kept iff sigma_i > max(dim) * eps * sigma_1, the standard
            # numerical-rank convention (same one torch/numpy matrix_rank uses internally).
            # reduce_merge's final re-expression step uses the SAME convention (its sum-of-
            # two-low-rank-matrices reconstruction is only lossless up to this same
            # numerical-rank threshold).
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
        # FIXED 2026-08-05 (collision confirmed in practice, context.md §8.7):
        # the split alone is not enough to disambiguate -- multiple DATASETS can
        # share the same (init_cls, increment) shape (e.g. cifar224/imagenetr/
        # omnibenchmark1k all use 10/10 in the wave1_final convention), which
        # previously made same-seed runs on different datasets silently
        # overwrite each other's reconstruction-error diagnostics. Dataset name
        # is now part of the tag; old filenames (pre-fix) are left untouched on
        # disk, this only changes what NEW runs write.
        # FIXED 2026-08-05 (cross-CAMPAIGN collision, not just cross-dataset):
        # even with dataset in the tag, two INDEPENDENT campaigns running
        # concurrently (different SLURM jobs, no shared coordination) still
        # collide whenever they happen to match on (dataset, eps, split, seed)
        # -- e.g. wave1_final and sketchlora_ablations_and_sens both running
        # SketchLoRA on ImageNet-R at eps=0.01 at the same time. A post-hoc
        # "write then move" can't fix this: the window between write and move
        # is exactly when the other job's write can land on the same path.
        # sketchlora_diag_dir (opt-in, config key) sidesteps the whole
        # tag-uniqueness problem: point each campaign at its OWN directory and
        # key the filename on this run's own `prefix` (already unique
        # project-wide by convention) instead of a descriptive tag that's only
        # unique WITHIN one campaign. Unset (every existing campaign) is
        # byte-identical to the old shared-path/tag-based behavior -- nothing
        # currently running or already collected is affected.
        diag_dir = args.get("sketchlora_diag_dir")
        if diag_dir is not None:
            prefix = args.get("prefix") or "sketchlora"
            self._diag_path = os.path.join(diag_dir, "sketchlora_diag_{}_seed{}.json".format(prefix, seed))
        else:
            dataset_tag = args.get("dataset", "unknown")
            if self.energy_target is not None:
                tag = "{}_adapt{}_b{}_{}".format(dataset_tag, self.energy_target, self.n_lora_blocks or "all", split)
            else:
                tag = "{}_r{}_b{}_{}".format(dataset_tag, self.svd_rank, self.n_lora_blocks or "all", split)
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
    # Each chunk trains residual slot(s) over the frozen sketch (slot 0); the
    # boundary folds (sketch (+) residuals) -> sketch via randomized SVD and resets
    # them, exactly like the per-task path but fired on the sample clock so eval
    # always sees the folded sketch. NOW period-aware (impl_plan_7.27.2026 sec 1.2 --
    # was previously "NOT period-aware, task-boundary only", guarded by an assert):
    # _stream_slot cycles through residual slots 1..P exactly like the oracle path's
    # _train_adapter, and _stream_end_chunk folds at period boundaries instead of
    # every cycle. svd_period defaults to 1, so the off-path (and any existing
    # config that never touched svd_period) is completely unaffected -- P=1 makes
    # every formula below reduce to the original single-slot-every-cycle behavior
    # exactly, same invariant the oracle path already documented.
    #
    # KNOWN LIMITATION (flagged, not solved this round): a trailing PARTIAL period
    # at the very end of a bounded-memory stream has no forced final fold (unlike
    # the oracle path's `_train`, which knows `_n_run_effective` and force-folds on
    # the last task). The harness does not expose "is this the last cycle" to the
    # model. This does NOT affect accuracy (eval/CIL forward sums sketch + every
    # trained residual slot via merge=True regardless of fold timing) -- it only
    # means persistent_state()/`_deployed_forward`'s end-of-run FLOPs/latency
    # measurement could slightly undercount if the very last period never closes.
    # Only matters for lazy_merge_mode="period"; plateau's max_holdoff cap bounds
    # the same risk to at most max_holdoff cycles and legacy_saturation/off fold
    # every cycle (P=1), so neither has this gap.
    def _stream_init(self):
        self._cur_task = -1                       # diagnostic chunk id used by _compress
        if self.classifier_alignment:
            self._ca_lazy_init_stats()

    def _ca_lazy_init_stats(self):
        """Build self._ca_stats once self._network.fc exists (not available at
        __init__ time -- update_fc runs later). Shared by both harnesses: the
        streaming path calls this from _stream_init (once, before any cycle);
        the oracle path (2026-08-05 fix -- classifier_alignment was previously
        DEAD CODE under trainer.py's plain per-task loop, since _stream_init is
        only ever called from bounded_memory_mixin.py/stream_mixin.py, never
        from models/lora.py::incremental_train -- see
        sketchlora_ablations_imagenetr20t's exactsvd_ca finding) calls this
        lazily from _train, guarded by `if self._ca_stats is None` so repeated
        per-task calls are a no-op after the first."""
        if self._ca_stats is None:
            from utils.ca import ClassStats
            net = self._network.module if hasattr(self._network, "module") else self._network
            self._ca_stats = ClassStats(feat_dim=net.fc.in_features, device=self._device,
                                         cov_mode=self.ca_cov_mode)

    def _ca_reset_reservoir(self):
        # (c) reset the real-feature reservoir at the START of each cycle, so
        # align_head (called at cycle end) only ever mixes in features that
        # genuinely came from THIS cycle's own training, never a stale
        # previous cycle's leftovers. Shared by both harnesses -- see
        # _ca_lazy_init_stats's docstring for why oracle mode needs its own
        # call site (models/lora.py::_train, via this class's own _train
        # override) rather than reusing _stream_begin_chunk.
        self._ca_real_buffer_feats = None
        self._ca_real_buffer_labels = None
        self._ca_real_buffer_n_seen = 0

    def _stream_slot(self):
        return RESIDUAL + (self._cur_task % self.svd_period)

    def _stream_begin_chunk(self, loader):
        # advance the chunk counter BEFORE training (not at chunk end) so
        # _stream_slot's modulo routing is correct DURING this chunk's training,
        # mirroring how the oracle path's self._cur_task is set before _train_adapter().
        self._cur_task += 1
        self._freeze_inactive_blocks()
        if self.ca_real_mix_frac > 0:
            self._ca_reset_reservoir()
        super()._stream_begin_chunk(loader)       # freeze_to_task(slot) + fresh optimizer

    def _stream_end_chunk(self, loader):
        self._freeze_inactive_blocks()
        if self.lazy_merge_mode in ("off", "period"):
            at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
            should_fold = at_period_boundary
        elif self.lazy_merge_mode == "legacy_saturation":
            # *** UNTESTED as of 2026-08-03 *** -- runs every cycle (whether or not
            # a fold fires this cycle), and per-module does a torch.linalg.svdvals
            # -- plan sec 4.1: "UNCOUNTED and expensive," the most costly of the
            # lazy-merge gates precisely because it runs unconditionally.
            with ce_region("sketchlora/lazy_saturation_check"):
                should_fold = self._lazy_should_fold()
        else:   # plateau
            # *** UNTESTED as of 2026-08-03 *** -- also runs every cycle
            # (plan sec 4.1). _residual_products() (building [d,d] per module)
            # is evaluated as part of this same statement, inside the region.
            with ce_region("sketchlora/lazy_plateau_check"):
                should_fold = self._plateau_tracker.should_fold(self._residual_products())

        self._last_cycle_folded = should_fold   # read by _ce_boundary_macs_this_cycle
        if should_fold:
            self._compress()                      # fold -> sketch, reset residual(s)
            if self.lazy_merge_mode == "plateau":
                self._plateau_tracker.reset()
        # else: accumulate -- leave the residual's (now further-trained) weights
        # exactly as they are; the next cycle's _stream_begin_chunk only
        # re-establishes trainability + a fresh optimizer, it does not touch
        # residual weights, so training continues on the SAME residual(s).

        # Classifier alignment (impl_plan_7.27.2026 sec 1.3): "after each fold (or
        # each cycle if no fold)" -- runs every cycle, independent of fold timing.
        if self.classifier_alignment:
            self._run_ca_alignment()

    def _run_ca_alignment(self):
        """Dispatches to whichever CA arm is configured. Shared by both
        harnesses (streaming's _stream_end_chunk calls this every cycle,
        unconditionally; the oracle path's _train override -- 2026-08-05 fix,
        see _ca_lazy_init_stats's docstring -- calls this every task, matching
        the SAME "every cycle" cadence since a task IS a cycle there).
        v2 (impl_plan_7.28.2026 sec 2): (d) logit_adjust_only is a STRUCTURALLY
        different arm (no head retraining, just an additive bias) -- dispatched
        separately, never combined with the ordinary align_head path."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        if self.ca_logit_adjust_only:
            from utils.ca import apply_logit_adjustment
            # *** UNTESTED as of 2026-08-03 *** -- the plan's formula-based
            # accounting (_ce_boundary_macs_this_cycle below) leaves this uncosted
            # by design ("no head retraining ... negligible against a real
            # forward/backward pass") -- tagged anyway (R5) since even a per-class
            # closed-form update has real, non-zero host-side Python-loop cost
            # (logit_adjust_bias's `for c in seen: ...`) that a MAC-only view
            # would never show regardless of how the formula treats it.
            with ce_region("sketchlora/ca_logit_adjust"):
                self._ca_logit_correction = apply_logit_adjustment(
                    net.fc, self._ca_stats, self.ca_logit_adjust_tau, self._ca_logit_correction)
            logging.info("[CA] cycle {}: logit_adjust_only tau={}".format(
                self._cur_task, self.ca_logit_adjust_tau))
        else:
            from utils.ca import align_head
            real_buf = None
            if self.ca_real_mix_frac > 0 and self._ca_real_buffer_feats is not None \
                    and self._ca_real_buffer_feats.shape[0] > 0:
                real_buf = (self._ca_real_buffer_feats, self._ca_real_buffer_labels)
            # *** UNTESTED as of 2026-08-03 *** -- boundary marker for align_head's
            # loop/loss/backward/optimizer cost specifically. align_head's own
            # internals (utils/ca.py) carry NESTED tags for the two sub-costs the
            # plan calls out as previously uncounted (ca_pseudo_feature_sampling,
            # ca_low_rank_factor_cache_build) -- per ce_profiler.py's EXCLUSIVE
            # attribution (a region's harvested cost stops at any nested ce/
            # scope), this outer tag's own number is "align_head minus those two,"
            # NOT a grand total -- the grand total is the SUM of this tag plus its
            # nested children, exactly what charged_macs()/charged_seconds()
            # already compute over the whole region dict.
            with ce_region("sketchlora/ca_alignment"):
                ca_result = align_head(
                    net.fc, self._ca_stats, self.ca_steps, self.ca_batch, self.ca_lr, self._device,
                    real_feature_buffer=real_buf, real_mix_frac=self.ca_real_mix_frac,
                    early_stop_patience=self.ca_early_stop_patience, val_batch_size=self.ca_val_batch)
            logging.info("[CA] cycle {}: steps={} final_loss={} stopped_early={}".format(
                self._cur_task, ca_result["steps"], ca_result["final_loss"],
                ca_result["stopped_early"]))

    def _ce_aux_macs_per_step(self):
        # impl_plan_7.27.2026 sec 2.3(a): sketch-inclusion forward overhead,
        # 2*d*r_hat MACs/token/module, GROWING over the stream as r_hat grows --
        # read the CURRENT mean rank from the diagnostics log (updated every
        # fold), not a fixed constant. 0 before the first fold has ever happened
        # (sketch slot still zero-width in effect).
        if not self._diag_records:
            return 0.0
        from utils.ce_formulas import sketchlora_step_macs_sketch_inclusion
        r_hat = self._diag_records[-1]["r_hat_mean"]
        if r_hat is None:
            return 0.0
        return sketchlora_step_macs_sketch_inclusion(r_hat)

    def _ce_boundary_macs_this_cycle(self, chunk_images, macs_per_image_fwd=0.0):
        # impl_plan_7.27.2026 sec 2.3(b/c): per-fold merge cost (only charged on
        # cycles that actually folded -- lazy-merge variants skip most cycles)
        # + CA alignment cost (charged every cycle CA runs, matching
        # _stream_end_chunk's own "every cycle" cadence for CA above).
        # macs_per_image_fwd unused here -- SketchLoRA's boundary costs (SVD
        # merge, CA alignment) don't involve extra full forward passes over
        # chunk-sized data, only accepted for signature compatibility with
        # the base hook (see models/stream_mixin.py).
        from utils.ce_formulas import sketchlora_fold_macs, sketchlora_ca_macs, N_MODULES
        out = {}
        if getattr(self, "_last_cycle_folded", False) and self._diag_records:
            r_hat = self._diag_records[-1]["r_hat_mean"] or 0.0
            out["fold_merge"] = sketchlora_fold_macs(r_hat, oversampling=self.oversampling,
                                                       merge_op=self.merge_op) * N_MODULES
        if self.classifier_alignment and not self.ca_logit_adjust_only:
            # (d) logit_adjust_only does no gradient-based training at all --
            # its cost is a per-class closed-form bias update, negligible
            # against a real forward/backward pass, so it's left uncosted here
            # (matching the plan's "no head retraining" framing) rather than
            # invented a MAC estimate for a handful of scalar log() calls.
            net = self._network.module if hasattr(self._network, "module") else self._network
            n_classes = net.fc.out_features
            out["ca_alignment"] = sketchlora_ca_macs(self.ca_steps, self.ca_batch, n_classes)
        return out

    def _residual_products(self):
        """[d,d] float B_r @ A_r per (block, {q,v}) x residual slot, in a FIXED,
        stable order across calls -- required by the plateau tracker's per-cycle
        element-wise comparison. (legacy_saturation's own check computes the same
        quantity inline and is left untouched to avoid any risk to its behavior.)"""
        products = []
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q), (attn.lora_A_v, attn.lora_B_v)):
                for slot in self._residual_slots():
                    A_r, B_r = A_list[slot].weight, B_list[slot].weight
                    products.append((B_r @ A_r).float())
        return products

    def _bounded_train_epoch(self, loader, optimizer, scheduler, cycle_class_mask, step_acc=None):
        """Identical to the base (bounded_memory_mixin.py) generic loop when
        classifier_alignment is off -- delegates straight to super(), so the
        off-path is bit-for-bit the base class's own method, not a reimplementation
        of it. When on, the SAME forward pass already computed for the training
        loss also yields "features" (no extra forward), fed to ClassStats."""
        if not self.classifier_alignment:
            return super()._bounded_train_epoch(loader, optimizer, scheduler, cycle_class_mask,
                                                step_acc=step_acc)
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            output = self._network(inputs, task=slot, merge=merge)
            logits = output["logits"]
            masked_logits = logits + cycle_class_mask
            loss = F.cross_entropy(masked_logits, targets)
            extra = run_step_narrow(step_acc, "sketchlora_extra",
                                    lambda: self._stream_extra_loss(0, logits.shape[1]))
            if not (isinstance(extra, float) and extra == 0.0):
                loss = loss + extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ClassStats.update is a per-SAMPLE Python loop with Welford
            # updates, and in cov_mode="shared_full" it also does a
            # [feat_dim,feat_dim] outer product per sample -- real, genuinely
            # per-step cost. Narrow-wrapped, epoch 0 only, same as elsewhere;
            # operates on output["features"], already computed above, so this
            # doesn't re-run any forward compute.
            def _ca_step_update(_feats=output["features"], _targets=targets):
                with ce_region("sketchlora/ca_class_stats_update"):
                    self._ca_stats.update(_feats, _targets)
                if self.ca_real_mix_frac > 0:
                    with ce_region("sketchlora/ca_reservoir_update"):
                        self._ca_buffer_update(_feats.detach(), _targets.detach())
            run_step_narrow(step_acc, "sketchlora_ca_step", _ca_step_update)
        if scheduler is not None:
            scheduler.step()

    @torch.no_grad()
    def _ca_buffer_update(self, features, labels):
        """(c) real-feature mixing: bounded reservoir of ACTUAL current-cycle
        features (classic reservoir sampling, uniform over everything seen
        this cycle so far) -- reset at the start of each cycle in
        _stream_begin_chunk so align_head only ever mixes in features that
        genuinely came from the chunk it's aligning after, never a stale
        cycle's leftovers."""
        cap = self.ca_real_reservoir_size
        if self._ca_real_buffer_feats is None:
            self._ca_real_buffer_feats = torch.zeros(0, features.shape[1], device=features.device)
            self._ca_real_buffer_labels = torch.zeros(0, dtype=labels.dtype, device=labels.device)
        for i in range(features.shape[0]):
            self._ca_real_buffer_n_seen += 1
            if self._ca_real_buffer_feats.shape[0] < cap:
                self._ca_real_buffer_feats = torch.cat(
                    [self._ca_real_buffer_feats, features[i:i + 1]], dim=0)
                self._ca_real_buffer_labels = torch.cat(
                    [self._ca_real_buffer_labels, labels[i:i + 1]], dim=0)
            else:
                j = torch.randint(0, self._ca_real_buffer_n_seen, (1,)).item()
                if j < cap:
                    self._ca_real_buffer_feats[j] = features[i]
                    self._ca_real_buffer_labels[j] = labels[i]

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

    # -- embedding-drift extraction (t-SNE study) ------------------------
    def _drift_test_loader(self, task_idx):
        known, total = self._task_class_ranges[task_idx]
        dataset = self.data_manager.get_dataset(np.arange(known, total), source="test", mode="test")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

    @torch.no_grad()
    def _extract_features(self, loader, forward_fn):
        """forward_fn(net, inputs) -> output dict with a "features" key. Returns
        (features, labels) as numpy arrays, in loader order (no shuffle)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        net.eval()
        feats, labels = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            out = forward_fn(net, inputs)
            feats.append(out["features"].cpu().numpy())
            labels.append(targets.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    def _save_drift_group(self, fname, features, labels, **meta):
        path = os.path.join(self.embed_drift_dir, fname)
        np.savez(path, features=features, labels=labels, **meta)
        logging.info("[embed_drift] wrote {} ({} points)".format(path, features.shape[0]))

    def _maybe_extract_drift_exact(self):
        """(a) the current task's own test images through the just-trained,
        uncompressed state -- sketch_{n-1} (frozen) + residual_n (trained),
        exactly what training used this task. Run BEFORE _compress()."""
        if self.embed_drift_dir is None:
            return
        n = self._cur_task
        loader = self._drift_test_loader(n)
        feats, labels = self._extract_features(
            loader, lambda net, x: net(x, task=self._train_adapter(), merge=self.train_merge))
        self._save_drift_group("exact_task{}.npz".format(n), feats, labels,
                                task=n, boundary=n, kind="exact")

    def _maybe_extract_drift_sketch(self):
        """(b) every task r <= n's test images through the sketch alone, right
        after this boundary's compression. Run AFTER _compress()."""
        if self.embed_drift_dir is None:
            return
        n = self._cur_task
        for r in range(n + 1):
            loader = self._drift_test_loader(r)
            feats, labels = self._extract_features(
                loader, lambda net, x: net(x, task=SKETCH, merge=True))
            self._save_drift_group("sketch_task{}_boundary{}.npz".format(r, n), feats, labels,
                                    task=r, boundary=n, kind="sketch")

    # -- train then compress, but ONLY at a period boundary (or the run's last task)
    # -- eval runs before after_task) --------------
    def _train(self, train_loader):
        self._task_class_ranges[self._cur_task] = (self._known_classes, self._total_classes)
        self._freeze_inactive_blocks()   # re-freeze before the optimiser is built
        # 2026-08-05 fix: classifier_alignment was DEAD CODE on this (oracle)
        # path -- its actual effect lived entirely in _stream_end_chunk, a
        # StreamMixin hook only ever invoked by bounded_memory_mixin.py's/
        # stream_mixin.py's own drivers, never by models/lora.py's plain
        # incremental_train->_train() call used here. _ca_stats was therefore
        # never even constructed and _run_ca_alignment never ran -- confirmed
        # by exactsvd_ca producing byte-identical accuracy curves to exactsvd
        # on every seed of sketchlora_ablations_imagenetr20t (see that
        # campaign's memory/results). Mirrors bounded_memory's own cadence: a
        # "cycle" there is a chunk; here a task IS the cycle (svd_period=1 in
        # every production/ablation config), so per-task is the correct analog
        # of "every cycle" for both the reservoir reset (start) and alignment
        # (end).
        if self.classifier_alignment:
            self._ca_lazy_init_stats()
            if self.ca_real_mix_frac > 0:
                self._ca_reset_reservoir()
            self._train_with_ca(train_loader)
        else:
            super()._train(train_loader)

        # R2 baseline-vs-actual, SELF-measured here (docs/ce_step_boundary_
        # isolation_plan.md sec 6.6), BEFORE _compress() runs. Fixes a real
        # timing bug: trainer.py's own generic post-incremental_train() R2 probe
        # (used by every other method, correctly -- see below) would read the
        # sketch's POST-fold state for SketchLoRA specifically, since
        # _compress() runs inside THIS method, before incremental_train()
        # returns -- i.e. it would measure the rank that's about to be used
        # NEXT task, not the rank that was actually in effect while the steps
        # that just ran were training. r_hat is constant for the whole task
        # (only _compress() changes it, and that hasn't happened yet at this
        # point), so one measurement here suffices -- no need to repeat this
        # every epoch. O-LoRA/InfLoRA/TreeLoRA don't need this: their fold
        # state (frozen_delta_q/v) is set by freeze_to_task()/fold_up_to()
        # BEFORE _train() runs, not after, so trainer.py's generic post-hoc
        # timing already reads the correct, in-effect-during-training state for
        # them (confirmed by tracing backbone/vit_lora.py::freeze_to_task).
        if getattr(self, "_ce_boundary_ctrl", None) is not None:
            self._ce_pre_boundary_probe(train_loader)

        at_period_boundary = (self._cur_task + 1) % self.svd_period == 0
        at_last_task = (self._cur_task + 1) >= self._n_run_effective
        if at_period_boundary or at_last_task:
            # oracle-mode boundary bookkeeping (docs/ce_step_boundary_isolation_
            # plan.md sec 7): under bounded_memory streaming this call site isn't
            # used -- _stream_end_chunk (above) is already wrapped end-to-end by
            # the driver's own boundary_end session.
            self._maybe_extract_drift_exact()
            with ce2_boundary(self):
                run_boundary(getattr(self, "_ce_boundary_ctrl", None), "sketchlora_compress",
                            self._compress)
            self._maybe_extract_drift_sketch()
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
        # Classifier alignment runs every task (== every cycle here), same
        # "after each fold (or each cycle if no fold)" semantics as
        # _stream_end_chunk -- independent of the compress gate above, placed
        # after it to match that method's structural ordering (align_head
        # operates purely on net.fc via _ca_stats, so the relative order vs.
        # compress has no functional effect either way).
        if self.classifier_alignment:
            with ce2_boundary(self):
                run_boundary(getattr(self, "_ce_boundary_ctrl", None), "sketchlora_ca",
                            self._run_ca_alignment)

    def _ce_pre_boundary_probe(self, train_loader):
        """R2 baseline-vs-actual, measured here rather than by trainer.py's
        generic fallback -- see the timing-bug explanation in _train() above.
        Single-batch, two profiled calls (measure_baseline_and_actual already
        keeps this cheap) -- run once per task, right before _compress(), never
        repeated within a task since r_hat can't change until the fold below."""
        from utils.ce_profiler import measure_baseline_and_actual
        _probe_inputs, _probe_targets = next(iter(train_loader))[1:]
        _probe_inputs = _probe_inputs.to(self._device)
        _probe_targets = _probe_targets.to(self._device)
        _lo, _hi = self._ce_slice()

        def _loss_fn(logits):
            return F.cross_entropy(logits[:, _lo:_hi], _probe_targets - _lo)

        baseline_fwd, baseline_bwd, actual_fwd, actual_bwd = measure_baseline_and_actual(
            self._network, _probe_inputs, _probe_targets, _loss_fn,
            self._train_adapter(), self.train_merge, self._device)
        self._network.zero_grad()
        self._ce_pre_boundary_r2 = (baseline_fwd, baseline_bwd, actual_fwd, actual_bwd)

    def _train_with_ca(self, train_loader):
        """Oracle-path counterpart to _bounded_train_epoch's CA-aware loop.
        Reimplements models/lora.py::_train's multi-epoch loop (rather than
        delegating to super()) so the SAME forward pass already computed for
        the training loss also yields "features" fed to ClassStats/the
        reservoir -- no extra forward, exactly matching _bounded_train_epoch's
        own design, just spanning every epoch of one task instead of a single
        streaming epoch."""
        self._network.to(self._device)
        params = self._optimizer_param_groups()
        optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.min_lr) if self.lr_anneal else None

        lo, hi = self._ce_slice()
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            # Step-type measurement (docs/ce_step_boundary_isolation_plan.md sec
            # 1a/7): CA's per-step bookkeeping operates on output["features"],
            # already computed by the forward pass above -- narrow-wrapping it
            # doesn't re-run any forward compute, just isolates the Welford
            # update/reservoir-sampling logic itself. Epoch 0 only, as elsewhere.
            step_acc = getattr(self, "_ce_step_acc", None) if epoch == 0 else None
            losses, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                output = self._network(inputs, task=self._train_adapter(), merge=self.train_merge)
                logits = output["logits"]
                local_logits = logits[:, lo:hi]
                local_targets = targets - lo
                loss = F.cross_entropy(local_logits, local_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                def _ca_step_update(_feats=output["features"], _targets=targets):
                    with ce_region("sketchlora/ca_class_stats_update"):
                        self._ca_stats.update(_feats, _targets)
                    if self.ca_real_mix_frac > 0:
                        with ce_region("sketchlora/ca_reservoir_update"):
                            self._ca_buffer_update(_feats.detach(), _targets.detach())
                run_step_narrow(step_acc, "sketchlora_ca_step", _ca_step_update)

                preds = local_logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)
            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            prog_bar.set_description(
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.epochs, losses / len(train_loader), train_acc))
        logging.info("Task {} finished. Train_accy {:.2f}".format(self._cur_task, train_acc))

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
        - ``reduce_merge``: inverts the order of every merge_op above (reduce THEN merge,
          not merge then reduce). Truncates the TRANSIENT residual's own spectrum first
          (randomized probe, same energy_target/svd_rank rule as randsvd/exactsvd applied
          to the residual alone), sums the truncated residual with the UNTOUCHED existing
          sketch, then re-expresses that sum via an EXACT (not randomized) SVD, keeping
          every direction above the numerical-rank threshold -- since the sum of two
          low-rank matrices is itself already low-rank (<= sketch rank + residual's
          truncated rank), this second step is a lossless re-expression, not a further
          compression. Ignores ``sketchlora_admission`` entirely (see the __init__
          warning) -- nothing is ever evicted from the existing sketch in this algorithm,
          only added to; ``sketchlora_rank_cap`` still applies as a hard ceiling.

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
        fd_rents = []   # per (layer,proj) FD-shrinkage stats this merge (fd_shrinkage only)
        floor_k_protected = []    # per (layer,proj) reserved-slot count actually used
        floor_energy_filled = [] # per (layer,proj) energy-filled slot count this merge
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
            (self.sketch_diag or self.energy_target is not None or self.fd_shrinkage)
        module_idx = 0     # unique per (layer,proj) -- seeds countsketch's hash/sign draw
        for attn in self._active_attns():
            for A_list, B_list in ((attn.lora_A_q, attn.lora_B_q),
                                   (attn.lora_A_v, attn.lora_B_v)):
                A_s, B_s = A_list[SKETCH].weight, B_list[SKETCH].weight     # [r,d],[d,r]
                dev, dt = B_s.device, B_s.dtype
                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: runs every
                # (block, proj) pair, every compress call, for every admission
                # rule (including "floor", which reuses this same delta_W) --
                # previously folded into one flat sketchlora_fold_macs formula.
                with ce_region("sketchlora/fold_composite_build"):
                    delta_W = B_s @ A_s                                    # [d, d], unscaled
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
                        # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1/R3: our
                        # own diagnostic bookkeeping, never charged to the method.
                        with ce_region("_excluded/sketch_diag"):
                            fro_delta = delta_W.float().norm()
                            retained.append(1.0)
                            sigma_next.append(0.0)
                            fro.append(fro_delta.item())
                            rhat.append(final_rank)
                    B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: nn.Linear
                    # construction + copy_ when the sketch slot's rank changes;
                    # previously UNCOUNTED entirely (fires on most adaptive-mode
                    # folds -- this skip_compression branch is only the FIRST-ever
                    # fold, but the same tag is reused in every branch below so
                    # they aggregate into one measured total).
                    with ce_region("sketchlora/fold_slot_realloc"):
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
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: kaiming/
                    # zeros re-init of every residual slot -- "small but real,"
                    # previously UNCOUNTED. Same tag reused in every branch below.
                    with ce_region("sketchlora/fold_residual_reset"):
                        for slot in residual_slots:
                            nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                            nn.init.zeros_(B_list[slot].weight)
                    module_idx += 1
                    continue

                if self.admission_rule == "floor":
                    # Structurally different from every other admission_rule: not a
                    # single top-r_hat_t truncation of the composite, but a
                    # protected-k + energy-filled construction (utils/admission.py).
                    # Fully self-contained (computes its own S, B_hat, A_hat,
                    # final_rank); everything AFTER this block (FD shrinkage,
                    # diagnostics, slot write-back) is shared/unchanged.
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: extra QR
                    # + full torch.linalg.svd(R_orth) + a second rand_svd inside
                    # floor_admission_merge (utils/admission.py) -- previously
                    # UNCOUNTED entirely (gated on admission_rule=="floor"). The
                    # R-composite build immediately above is specific to this
                    # admission rule (not shared with fold_composite_build's
                    # delta_W), so it is included in the same tag.
                    with ce_region("sketchlora/fold_merge_admission_floor"):
                        R = None
                        for slot in residual_slots:
                            A_r, B_r = A_list[slot].weight, B_list[slot].weight
                            term = (B_r @ A_r).float()
                            R = term if R is None else R + term
                        residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                        B_hat, A_hat, final_rank, S, g_stats = floor_admission_merge(
                            delta_W.float(), B_s.float(), R, residual_total,
                            self.energy_target, self.admission_floor_k, self.rank_cap,
                            self.oversampling, rand_svd)
                        B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                    floor_k_protected.append(g_stats["k_protected"])
                    floor_energy_filled.append(g_stats["energy_filled"])
                    if self.sketch_diag:
                        with ce_region("_excluded/sketch_diag"):
                            recon_err = (delta_W.float() - (B_hat.float() @ A_hat.float())).norm()
                            fro_delta = delta_W.float().norm()
                            retained.append(1.0 - (recon_err / fro_delta).item() ** 2 if fro_delta > 0 else 1.0)
                            sigma_next.append(S[final_rank].item() if S.numel() > final_rank else 0.0)
                            fro.append(fro_delta.item())
                            rhat.append(final_rank)
                    with ce_region("sketchlora/fold_residual_reset"):
                        for slot in residual_slots:
                            nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                            nn.init.zeros_(B_list[slot].weight)
                    with ce_region("sketchlora/fold_slot_realloc"):
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
                    module_idx += 1
                    continue

                if self.merge_op == "reduce_merge":
                    # "Reduce THEN merge" ablation (2026-08-05 user request), inverting
                    # every other merge_op's order (merge then reduce): truncate the
                    # TRANSIENT per-task update's own spectrum BEFORE it ever combines
                    # with the sketch, then re-express the sum losslessly (exact SVD,
                    # no further truncation) instead of truncating the combined sum the
                    # way randsvd/exactsvd do. Self-contained (own S/B_hat/A_hat/
                    # final_rank), falls through to the shared diagnostics/slot-realloc/
                    # residual-reset tail below, same pattern as the admission_rule==
                    # "floor" branch above. admission_rule is ignored here (see the
                    # __init__ warning) -- there is no eviction step in this algorithm,
                    # only addition, so "how much of the existing sketch to evict"
                    # never arises.
                    with ce_region("sketchlora/fold_merge_reduce_merge"):
                        # 1-3: reduce the residual ALONE, before it sees the sketch --
                        # sum just the residual slots (NOT B_s@A_s), randomized-probe-
                        # decompose, truncate by the SAME energy_target/svd_rank rule
                        # every other adaptive merge_op uses, just applied to the
                        # residual's own spectrum instead of the full composite's.
                        R = None
                        for slot in residual_slots:
                            A_r, B_r = A_list[slot].weight, B_list[slot].weight
                            term = (B_r @ A_r).float()
                            R = term if R is None else R + term
                        residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                        U_r, S_r, Vh_r = rand_svd_probe(R, residual_total, self.oversampling)
                        if self.energy_target is not None:
                            total_r = S_r.pow(2).sum()
                            if total_r > 0:
                                cum_r = torch.cumsum(S_r.pow(2), 0) / total_r
                                r_low = int((cum_r < (1.0 - self.energy_target)).sum().item()) + 1
                            else:
                                r_low = 1
                            r_low = max(1, min(r_low, residual_total))
                        else:
                            r_low = min(self.svd_rank, residual_total)

                        # 4-5 COLLAPSED (user simplification, 2026-08-05): B_low @ A_low
                        # is exactly U_r[:,:r_low] @ diag(S_r[:r_low]) @ Vh_r[:r_low,:] --
                        # the standard truncated-SVD reconstruction formula -- so there's
                        # no need to materialize B_low/A_low as separate LoRA factors
                        # just to immediately multiply them back together; build the
                        # [d,d] product directly from the probe's U/S/Vh instead.
                        reduced_residual = U_r[:, :r_low] @ (S_r[:r_low].unsqueeze(1) * Vh_r[:r_low, :])

                        # 6: sum with the UNTOUCHED existing sketch -- nothing evicted,
                        # only added.
                        merged = (B_s.float() @ A_s.float()) + reduced_residual

                        # 7-8: EXACT (not randomized) SVD of the sum, re-expressed
                        # WITHOUT further truncation -- merged's TRUE rank is already
                        # <= r_hat(sketch) + r_low (a sum of two low-rank matrices), so
                        # keeping every direction above the standard numerical-rank
                        # threshold (same nocompress_eps convention "nocompress" already
                        # uses) is a LOSSLESS re-expression, not a third round of
                        # compression. rank_cap still applies as a hard ceiling (A5.1's
                        # plain clamp, same as every other merge_op) so this can't grow
                        # unboundedly if r_low doesn't shrink fast enough on its own.
                        U, S, Vh = torch.linalg.svd(merged)
                        thresh = max(merged.shape) * self.nocompress_eps * (S[0].item() if S.numel() else 0.0)
                        final_rank = max(1, int((S > thresh).sum().item()))
                        if self.rank_cap is not None:
                            final_rank = min(final_rank, self.rank_cap)
                        root_S = S[:final_rank].sqrt()
                        B_hat = (U[:, :final_rank] * root_S.unsqueeze(0)).to(dt)
                        A_hat = (root_S.unsqueeze(1) * Vh[:final_rank, :]).to(dt)
                    B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)

                    if self.sketch_diag:
                        with ce_region("_excluded/sketch_diag"):
                            recon_err = (delta_W.float() - (B_hat.float() @ A_hat.float())).norm()
                            fro_delta = delta_W.float().norm()
                            retained.append(1.0 - (recon_err / fro_delta).item() ** 2 if fro_delta > 0 else 1.0)
                            sigma_next.append(S[final_rank].item() if S.numel() > final_rank else 0.0)
                            fro.append(fro_delta.item())
                            rhat.append(final_rank)

                    with ce_region("sketchlora/fold_slot_realloc"):
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
                    with ce_region("sketchlora/fold_residual_reset"):
                        for slot in residual_slots:
                            nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                            nn.init.zeros_(B_list[slot].weight)
                    module_idx += 1
                    continue

                S = None
                U_probe = Vh_probe = None   # populated only for merge_op=="randsvd" -- see below
                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1 "fold_rank_select":
                # the svdvals/full-svd spectrum computation AND the subsequent
                # r_hat_t decision-tree (bounded_eviction/cap/energy_target/
                # nocompress/naive_sum/countsketch) -- both genuinely part of
                # "how the rank gets chosen," previously folded into one flat
                # sketchlora_fold_macs formula regardless of which admission path
                # actually ran this cycle.
                #
                # FIXED 2026-08-03 (user-flagged, "catastrophic"): for
                # merge_op=="randsvd" this used to call torch.linalg.svdvals on
                # the FULL exact delta_W here, then rand_svd() (its own SEPARATE
                # randomized decomposition) later to build the factors -- the
                # rank decision was using perfect knowledge of the true spectrum
                # a randomized method should never have, and every fold paid for
                # two unrelated SVDs instead of one. Fixed: when merge_op==
                # "randsvd", run rand_svd_probe ONCE here (the SAME randomized
                # projection + small-matrix-SVD rand_svd() does internally),
                # sized to a working_rank that is an EXACT upper bound on
                # delta_W's true rank (prev_rank + residual_total -- derivable
                # from the LoRA structure itself, not estimated); the rank
                # decision below now reads THIS randomized S, and the
                # construction phase further down slices the SAME U_probe/S/
                # Vh_probe instead of calling rand_svd a second time.
                prev_rank = A_s.shape[0]
                residual_total = sum(A_list[slot].weight.shape[0] for slot in residual_slots)
                composite_rank = prev_rank + residual_total
                with ce_region("sketchlora/fold_rank_select"):
                    if full_svd_needed:
                        U, S, Vh = torch.linalg.svd(delta_W.float())          # reused for diag + recon
                    elif self.merge_op == "randsvd" and need_svdvals:
                        # fixed-rank randsvd (energy_target is None) still needs
                        # enough columns for the eventual self.svd_rank slice,
                        # which composite_rank alone is not guaranteed to cover
                        # (e.g. early in training, before enough has accumulated).
                        _working_rank = (composite_rank if self.energy_target is not None
                                        else max(composite_rank, self.svd_rank))
                        U_probe, S, Vh_probe = rand_svd_probe(delta_W, _working_rank, self.oversampling)
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
                            # (The floor variant of this formula -- never evict more than
                            # residual_total - k of the new directions -- was tried as
                            # "force_increase" and RETIRED 2026-07-28: its at-cap branch
                            # below ignored the floor entirely, so it silently degenerated
                            # to plain bounded_eviction once rank hit the cap. Superseded by
                            # admission_rule="floor", a separate top-level dispatch above
                            # that fixes this by construction -- see utils/admission.py.)
                            # prev_rank/residual_total/composite_rank now computed once,
                            # unconditionally, above (needed there for rand_svd_probe's
                            # working_rank too) -- reused here rather than recomputed.
                            cap = self.rank_cap if self.rank_cap is not None else composite_rank
                            if composite_rank > cap:
                                # at/above the cap: evict exactly enough to return to r_max.
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
                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: each merge_op
                # gets its OWN region name (not one shared "fold_merge" tag) --
                # only one branch runs per cycle (R4), and distinct names let the
                # ledger show which merge algorithm is actually driving the cost
                # on ablation sweeps over merge_op, rather than a single number
                # that silently means something different depending on config.
                if self.merge_op == "naive_sum":
                    # no SVD at all: literal running sum of the raw factor matrices
                    # (Experiments_Timeline.pdf sec 1.b.iii.3) -- does NOT preserve
                    # delta_W = B_hat @ A_hat in general; that is the deliberate point of this
                    # ablation, not a bug.
                    with ce_region("sketchlora/fold_merge_naive_sum"):
                        B_hat, A_hat = B_s.clone(), A_s.clone()
                        for slot in residual_slots:
                            B_hat = B_hat + B_list[slot].weight
                            A_hat = A_hat + A_list[slot].weight
                elif self.merge_op == "countsketch":
                    with ce_region("sketchlora/fold_merge_countsketch"):
                        B_ws = [B_s] + [B_list[slot].weight for slot in residual_slots]
                        A_ws = [A_s] + [A_list[slot].weight for slot in residual_slots]
                        seed = (int(self.cs_seed) * 1_000_003 + (self._cur_task + 1) * 9176
                                + module_idx) % (2 ** 63 - 1)
                        B_hat, A_hat = countsketch_compress(B_ws, A_ws, r_hat_t, seed)
                elif full_svd_needed:
                    # plan sec 4.1 names this "fold_merge_randsvd"'s sibling for the
                    # exactsvd/nocompress merge_ops -- reconstruction from the S/U/Vh
                    # already computed in fold_rank_select above (no second SVD).
                    with ce_region("sketchlora/fold_merge_full_svd_reconstruct"):
                        root_S = S[:r_hat_t].sqrt()
                        B_hat = U[:, :r_hat_t].to(dt) * root_S.to(dt).unsqueeze(0)
                        A_hat = root_S.to(dt).unsqueeze(1) * Vh[:r_hat_t, :].to(dt)
                else:
                    with ce_region("sketchlora/fold_merge_randsvd"):
                        # FIXED 2026-08-03 (see fold_rank_select above): if a probe
                        # decomposition was already computed there (the normal case
                        # whenever need_svdvals is True -- i.e. sketch_diag,
                        # energy_target, or fd_shrinkage is on, which is every
                        # production config), reuse it -- slicing to r_hat_t is the
                        # ONLY thing that happens here, no second SVD. Falls back to
                        # a single fresh rand_svd() call only in the narrow case
                        # where none of those three were on (need_svdvals was False,
                        # so fold_rank_select never ran a decomposition at all) --
                        # still exactly one SVD either way, never two.
                        if U_probe is not None:
                            B_hat, A_hat = factors_from_probe(U_probe, S, Vh_probe, r_hat_t)
                        else:
                            # gesvd fallback path (utils/randsvd.py, 2026-07-28) is
                            # inside rand_svd itself -- this tag covers it
                            # automatically whenever it fires.
                            B_hat, A_hat = rand_svd(delta_W, r_hat_t, self.oversampling)
                B_hat, A_hat = B_hat.to(dev, dt), A_hat.to(dev, dt)
                final_rank = B_hat.shape[1]

                # -- FD shrinkage (impl_plan_7.27.2026 sec 1.1): AFTER the eviction
                # count/rank l is chosen and the composite is truncated, BEFORE
                # diagnostics (so retained-energy below reflects the post-shrink
                # state -- shrinkage deliberately trades reconstruction fidelity for
                # bounding growth, the intended effect). Scoped to randsvd/exactsvd
                # (the two truncated-SVD merge_ops); no-op (with a startup warning
                # already logged) otherwise, since "Sigma[l]" has no meaning for a
                # hash-based or literal-sum merge.
                if self.fd_shrinkage and self.merge_op in ("randsvd", "exactsvd") and S is not None:
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1: previously
                    # UNCOUNTED entirely (gated on fd_shrinkage=True).
                    with ce_region("sketchlora/fold_fd_shrinkage"):
                        from utils.fd import apply_fd_shrinkage
                        B_hat, A_hat, fd_stats = apply_fd_shrinkage(B_hat, A_hat, S, final_rank)
                        fd_rents.append(fd_stats)
                        while len(self._fd_cumulative_rent) <= module_idx:
                            self._fd_cumulative_rent.append(0.0)
                        self._fd_cumulative_rent[module_idx] += fd_stats["rent"]

                if self.sketch_diag:
                    # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1/R3: our own
                    # diagnostic bookkeeping (recon-error measurement), never
                    # charged to the method.
                    with ce_region("_excluded/sketch_diag"):
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

                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1 "fold_slot_realloc":
                # same tag as the skip_compression/floor branches above -- fires on
                # most adaptive-mode folds, previously UNCOUNTED.
                with ce_region("sketchlora/fold_slot_realloc"):
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
                # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.1 "fold_residual_reset":
                # same tag as the skip_compression/floor branches above.
                with ce_region("sketchlora/fold_residual_reset"):
                    for slot in residual_slots:
                        nn.init.kaiming_uniform_(A_list[slot].weight, a=math.sqrt(5))
                        nn.init.zeros_(B_list[slot].weight)
                module_idx += 1
        self._sketch_populated = True
        if self.sketch_diag:
            self._record_diag(retained, sigma_next, fro, rhat, fd_rents,
                               floor_k_protected, floor_energy_filled)

    def _record_diag(self, retained, sigma_next, fro, rhat, fd_rents=None,
                      floor_k_protected=None, floor_energy_filled=None):
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
        if fd_rents:
            # impl_plan_7.27.2026 sec 1.1: pre/post-shrink total energy, rent
            # (=Sigma[l]^2 charged per kept direction), cumulative rent per module.
            rec["fd_pre_shrink_energy"] = [r["pre_shrink_energy"] for r in fd_rents]
            rec["fd_post_shrink_energy"] = [r["post_shrink_energy"] for r in fd_rents]
            rec["fd_rent"] = [r["rent"] for r in fd_rents]
            rec["fd_cumulative_rent"] = list(self._fd_cumulative_rent)
        if floor_k_protected:
            # 2026-07-28 guaranteed-admission direction: how many of the k reserved
            # slots actually had a nonzero orthogonal direction to admit this merge,
            # and how many additional slots the energy-fill step contributed.
            rec["floor_k_protected"] = floor_k_protected
            rec["floor_energy_filled"] = floor_energy_filled
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
