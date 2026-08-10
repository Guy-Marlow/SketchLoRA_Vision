"""Plan C — bounded-working-memory, boundary-free streaming for LAMDA-PILOT.

See impl_plan_7.25.2026/plan_C_task_agnostic.md §C1 for the full specification
and docs/plan_c_bounded_memory_harness.md for the implementation writeup. This
is a SEPARATE, additive codepath (boundary_mode: "bounded_memory") from the
existing sample/budget/sample_legacy streaming designs in models/stream_mixin.py
-- none of those files are modified by this one. It reuses the per-method hooks
already defined by StreamMixin (_stream_init/_stream_slot/_stream_train_merge/
_stream_begin_chunk/_stream_end_chunk/_stream_extra_loss/_stream_cil_forward),
so every method's existing bookkeeping override (O-LoRA's new slot + orthogonality,
InfLoRA's DualGPM + new slot, SketchLoRA's compress, TreeLoRA's tree update,
SeqLoRA's no-op) works here UNCHANGED -- nothing in models/olora.py, inflora.py,
treelora.py, seqlora.py needed to change for this harness to work.

What's genuinely different from stream_run() (models/stream_mixin.py), per Plan
C §C1, and why each needs its own new code rather than reusing stream_run()'s:

  1. Classifier head is pre-built to the FULL label space ONCE before training
     and never grows again (stream_run grows it incrementally via update_fc as
     new classes enter each chunk -- itself a structural task-boundary signal
     Plan C wants removed).
  2. ROUND 2 REVISION (impl_plan_7.26.2026/bounded_memory_round2_plan.md §1.1):
     round-1's unmasked full-1000-way cross-entropy is RETRACTED -- it made
     every absent class (including every previously-learned one) a permanent
     negative every single step, a known logit-suppression failure mode (cf.
     ACE, Caccia et al.), and is the primary suspect for round-1's ~30pt
     first-checkpoint deficit and all-method convergence. Loss is now masked
     to the union of classes present in the CURRENT CYCLE's own training data
     (cycle-mask, chosen over the ACE-style "mask only classes new to the
     stream" alternative for uniformity with every other local-CE convention
     already in this codebase) -- still no [lo,hi) contiguous-range slicing,
     an arbitrary per-cycle class SET via an additive -inf logit mask instead,
     since a cycle's classes need not be contiguous. Still boundary-blind: the
     mask is built from the cycle's own raw targets, never from a task index.
     ALL FOUR round-1 bounded_memory results (100MB/50T, 50MB/15T, 150MB/30T,
     200MB/30T) are retired to diagnostic-only status by this fix -- see
     docs/plan_c_bounded_memory_round2.md.
  3. Eval fires on DATA VOLUME checkpoints (fixed fractions of total stream
     length), not on real-task completions (stream_run's _stream_eval fires
     exactly when a chunk's cumulative image count reaches a real task's own
     cumulative image count -- i.e. still task-completion-pinned).
  4. The memory budget B is specified as a flat MB count (args["bm_budget_mb"],
     the closest analogs to Plan C §C2's fractional spec are {50, 75, 100,
     200}MB -- USER OVERRIDE 2026-07-25: flat MB, not a fraction of mean latent
     task size, per explicit instruction superseding the plan's own {0.2x,
     0.5x, 1x, 2x}-of-mean-task-size framing), using the same 224x224x3
     bytes/image convention as stream_mixin.py/budget_stream.py.
  5. InfLoRA's total-session count T is a concession value computed from the
     cycle structure (ceil(stream_images / cycle_images)), not the real task
     count -- set once via _bounded_set_total_sessions before training starts.

Leak audit (Plan C §C1, "blocking"): grep this file for any read of
self._cur_task / task index / real-task boundary INSIDE the training or
eval-windowing logic -- there is none. The only per-real-task bookkeeping is
`task_cumends` / the `_nearest_latent_task` field written into each result
dict, which is write-only telemetry for offline analysis figures (passed to
nothing that affects training, gradients, head growth, or checkpoint timing)
-- see docs/plan_c_bounded_memory_harness.md's leak-audit section for the full
accounting against Plan A §A4.3's eval-routing asserts.
"""

import hashlib
import json
import logging
import math
import os

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.metrics_logger import MetricsLogger
from utils.ops_ledger import OpsLedger, measure_step_macs
from utils.ce_profiler import (CEProfileController, measure_baseline_and_actual,
                                NarrowAuxAccumulator, split_by_recurrence, run_step_narrow)

# 2026-08-10: 8->4, see models/lora.py's identical change for rationale.
num_workers = 4
BYTES_PER_IMAGE = 224 * 224 * 3   # same accounting convention as budget_stream.py / stream_mixin.py


class BoundedMemoryMixin:
    # -- per-method hook, default = no-op (most methods don't need T) -------
    def _bounded_set_total_sessions(self, total_sessions):
        """Plan C §C1 InfLoRA concession: total-session count T, computed a
        priori from the cycle structure (ceil(stream_images/cycle_images)),
        fed to whichever method's ramp-style hyperparameter needs a total-count
        denominator (currently only InfLoRA's DualGPM threshold ramp). Default
        no-op; InfLoRA overrides this to set self.total_sessions."""
        pass

    @torch.no_grad()
    def _bounded_param_hash(self):
        """Round 2 §2.2: eval-routing identity check, extended to volume
        checkpoints per Plan A §A4.3 ("every eval checkpoint asserts + logs
        the parameter-state hash it evaluates"). Bounded-memory checkpoints
        are, by construction, always on a clock independent of any task/chunk
        boundary -- exactly the "checkpoint/task indices driven apart"
        scenario §A4.3 exists to guard, since a bug that made eval silently
        read a stale/frozen state instead of the just-trained one would
        otherwise be invisible from the accuracy numbers alone. Hashes every
        named parameter's bytes (order-independent via sorted names); called
        once per checkpoint in bounded_memory_run, which also asserts
        consecutive checkpoints' hashes differ (training between checkpoints
        must have changed SOMETHING)."""
        net = self._network.module if hasattr(self._network, "module") else self._network
        h = hashlib.sha256()
        for name, p in sorted(net.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    @torch.no_grad()
    def _bounded_eval(self, all_data, all_targets, cum_images, data_manager, task_class_cumends=None):
        """CIL-only eval (Plan C §C1: TIL is not computed -- no task identities
        exist in this setting). Mask logits to classes seen so far, where
        "seen" is derived purely from STREAM POSITION (max class index among
        the first `cum_images` stream entries), never from a task/chunk index.

        ADDED 2026-07-25: also returns a per-latent-task accuracy breakdown
        (the forgetting curve -- accuracy on each real task's OWN classes,
        measured at this checkpoint) alongside the pooled top-1 CIL number.
        This is ANALYSIS-ONLY: task_class_cumends is used purely to slice the
        test set / predictions for reporting after the CIL forward pass is
        already computed exactly as before; it does not change the model's
        forward, the loss, the head, or which checkpoint fires (leak-audit
        consistent -- see module docstring). If task_class_cumends is None,
        per-task breakdown is skipped (returns None) -- kept optional so any
        external caller of this method with the old 4-arg signature still
        works, though bounded_memory_run itself always passes it now.

        ADDED 2026-07-25 (later): top-5 accuracy alongside top-1, both pooled
        and per-task -- same single forward pass, just also keeping the top-5
        logit indices per batch (topk over the same masked logits used for
        top-1), so this costs one extra topk call, not an extra eval pass."""
        self._network.eval()
        hi_total = int(all_targets[:cum_images].max()) + 1
        k5 = min(5, hi_total)
        test_dataset = data_manager.get_dataset(
            np.arange(0, hi_total), source="test", mode="test")
        loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
        correct, correct5, n = 0, 0, 0
        per_task_correct, per_task_correct5, per_task_n = {}, {}, {}
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            logits = self._stream_cil_forward(inputs)["logits"][:, :hi_total]
            pred = logits.argmax(1).cpu().numpy()
            top5 = torch.topk(logits, k=k5, dim=1, largest=True, sorted=True)[1].cpu().numpy()
            tgt = targets.numpy()
            hit1 = pred == tgt
            hit5 = (top5 == tgt[:, None]).any(axis=1)
            correct += int(hit1.sum())
            correct5 += int(hit5.sum())
            n += len(tgt)
            if task_class_cumends is not None:
                task_ids = np.searchsorted(task_class_cumends, tgt, side="right")
                for tid in np.unique(task_ids):
                    m = task_ids == tid
                    per_task_correct[int(tid)] = per_task_correct.get(int(tid), 0) + int(hit1[m].sum())
                    per_task_correct5[int(tid)] = per_task_correct5.get(int(tid), 0) + int(hit5[m].sum())
                    per_task_n[int(tid)] = per_task_n.get(int(tid), 0) + int(m.sum())
        acc = round(100.0 * correct / max(n, 1), 2)
        acc5 = round(100.0 * correct5 / max(n, 1), 2)
        per_task_acc, per_task_acc5 = None, None
        if task_class_cumends is not None:
            n_tasks_seen = max(per_task_correct.keys(), default=-1) + 1
            per_task_acc = [
                round(100.0 * per_task_correct.get(t, 0) / max(per_task_n.get(t, 0), 1), 2)
                for t in range(n_tasks_seen)
            ]
            per_task_acc5 = [
                round(100.0 * per_task_correct5.get(t, 0) / max(per_task_n.get(t, 0), 1), 2)
                for t in range(n_tasks_seen)
            ]
        return acc, acc5, hi_total, per_task_acc, per_task_acc5

    def _bounded_new_optimizer(self):
        """Round-2 §1.2: HEAD weight_decay=0, uniformly across every method,
        for the bounded_memory path specifically. Overwrites whatever
        self._stream_optim/_stream_sched _stream_begin_chunk's own call to
        _stream_new_optimizer() (models/stream_mixin.py) just built there --
        that shared function is left completely untouched (still used
        unmodified by stream_run/sample_legacy), so this only ever affects
        bounded_memory runs, and never retroactively changes what any
        already-completed stream_run result reflects.

        Reuses self._optimizer_param_groups() (models/lora.py, overridden by
        SketchLoRA for its LoRA-wd=0 split) as the base grouping, then pulls
        the classifier head out into its own zero-weight-decay group. NOTE:
        _stream_new_optimizer() never consulted _optimizer_param_groups() at
        all (it always built one flat, uniform-weight_decay param list) --
        so SketchLoRA's lora_wd=0 setting was silently inert under EVERY
        streaming run before this fix (both stream_run and bounded_memory);
        it only ever took effect under the ordinary per-task training loop.
        This method is the first place that setting actually applies under
        streaming, and only for bounded_memory going forward."""
        base = self._optimizer_param_groups()
        head_ids = {id(p) for p in self._network.fc.parameters()}
        head_params = list(self._network.fc.parameters())
        if base and isinstance(base[0], dict):
            groups = []
            for g in base:
                remaining = [p for p in g["params"] if id(p) not in head_ids]
                if remaining:
                    groups.append({"params": remaining, "weight_decay": g["weight_decay"]})
        else:
            groups = [{"params": [p for p in base if id(p) not in head_ids],
                       "weight_decay": self.weight_decay}]
        groups.append({"params": head_params, "weight_decay": 0.0})
        self._stream_optim = optim.AdamW(groups, lr=self.init_lr)
        self._stream_sched = optim.lr_scheduler.CosineAnnealingLR(
            self._stream_optim, T_max=self.epochs, eta_min=self.min_lr) \
            if getattr(self, "lr_anneal", True) else None

    def _bounded_train_epoch(self, loader, optimizer, scheduler, cycle_class_mask, step_acc=None):
        """One epoch over a FIXED memory cycle's contents. Round-2 §1.1 fix:
        loss is now MASKED cross-entropy, restricted to the union of classes
        actually present in THIS CYCLE's own training data (cycle_class_mask,
        precomputed once per cycle in bounded_memory_run from the cycle's own
        raw targets -- purely data-derived, boundary-blind: nothing about
        real task identity is read to build it). Implementation: additive
        -inf mask on the classes NOT in the cycle, applied to the full-width
        logits before cross_entropy -- targets stay full-width class indices,
        no remapping needed, and the masked-out classes contribute exactly
        zero probability mass (exp(-inf)=0), so this is mathematically
        equivalent to slicing to just the cycle's classes without the index
        bookkeeping a slice would need. Chosen over the ACE-style alternative
        (mask only classes never before seen in the stream) for uniformity
        with every other local-CE convention already in this codebase
        (stream_run, budget_stream, the plain per-task loop all mask to the
        current batch/chunk's own class content) -- see docs/plan_c_
        bounded_memory_harness.md for the recorded choice.

        step_acc: 2026-08-05 addition (docs/ce_step_boundary_isolation_plan.md
        sec 1a/7/8) -- an optional NarrowAuxAccumulator, passed by the driver
        only for epoch 0 of a profiled cycle (None every other epoch/cycle).
        Only the isolated _stream_extra_loss() call is narrow-wrapped -- NOT
        the surrounding forward/backward/optimizer.step() -- which is what
        keeps profiling literally every step of epoch 0 affordable (the driver
        used to wrap this entire method instead, tracing the whole ViT step
        just to extract a few small matmuls' worth of aux cost)."""
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=slot, merge=merge)["logits"]
            masked_logits = logits + cycle_class_mask
            loss = F.cross_entropy(masked_logits, targets)
            extra = run_step_narrow(step_acc, "bounded_step_extra",
                                    lambda: self._stream_extra_loss(0, logits.shape[1]))
            if not (isinstance(extra, float) and extra == 0.0):
                loss = loss + extra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def bounded_memory_run(self, data_manager, args):
        """Plan C §C1 driver. Returns the same list-of-dict shape as
        stream_run() (`{"completed_frac", "cil", ...}` per checkpoint) so
        trainer.py's reporting/persistence code (_run_bounded_memory) can reuse
        the exact same pattern as _run_stream."""
        self.data_manager = data_manager
        epochs = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[bounded_mem] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        # Flat MB budget (USER OVERRIDE 2026-07-25, see module docstring item 4) --
        # closest analogs to Plan C §C2's {0.2x,0.5x,1x,2x}-of-mean-task-size spec
        # are {50, 75, 100, 200}MB; required, no silent default.
        budget_mb = float(args["bm_budget_mb"])
        seed_arg = args.get("seed", 1993)
        seed0 = seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg

        # ---- precompute the fixed, non-repeating, task-major unique-image
        # stream (identical construction to stream_run(), copied rather than
        # imported to keep this codepath fully independent/reversible) ----
        data_parts, targets_parts = [], []
        task_class_sizes = []   # classes per task (for class->task lookup, eval breakdown)
        task_image_sizes = []   # IMAGES per task (for stream-position->task lookup, logging only)
        known = 0
        for t in range(_n_run):
            task_size = data_manager.get_task_size(t)   # classes in this task
            task_class_sizes.append(task_size)
            lo, hi = known, known + task_size
            task_data, task_targets, _ = data_manager.get_dataset(
                np.arange(lo, hi), source="train", mode="train", ret_data=True)
            task_image_sizes.append(len(task_data))
            perm = np.random.RandomState(seed0 * 9973 + t).permutation(len(task_data))
            data_parts.append(task_data[perm])
            targets_parts.append(task_targets[perm])
            known = hi
        all_data = np.concatenate(data_parts)
        all_targets = np.concatenate(targets_parts)
        total_images = len(all_targets)
        mean_task_images = total_images / _n_run   # logging/context only, not used in the budget calc

        cycle_images = max(1, round(budget_mb * 1024 * 1024 / BYTES_PER_IMAGE))
        total_sessions = math.ceil(total_images / cycle_images)   # InfLoRA T concession
        self._bounded_set_total_sessions(total_sessions)
        # exposed for trainer.py::_run_bounded_memory's final write, so its
        # metadata matches the incremental writes below exactly
        self._bounded_total_images = total_images
        self._bounded_cycle_images = cycle_images
        self._bounded_total_sessions = total_sessions

        # ---- eval checkpoint schedule: fixed fractions of TOTAL STREAM
        # LENGTH, data-derived, computed once up front -- never adjusted by
        # anything that happens during training (Plan C §C1: "eval checkpoints
        # fire on DATA VOLUME"). Omni-1K: every 10% (10 points); everything
        # else: every 5% (20 points). Plus final. ----
        step = 0.10 if args["dataset"] == "omnibenchmark1k" else 0.05
        fractions = [round(f, 2) for f in np.arange(step, 1.0 + 1e-9, step)]
        if fractions[-1] != 1.0:
            fractions.append(1.0)
        checkpoint_images = sorted(set(max(1, round(f * total_images)) for f in fractions))

        logging.info(
            "[bounded_mem] budget_mb={} -> cycle={} images (mean_task={:.1f} images, "
            "so this budget is {:.2f}x mean task size); {} total images -> {} cycles "
            "(T={} sessions); {} eval checkpoints".format(
                budget_mb, cycle_images, mean_task_images, cycle_images / mean_task_images,
                total_images, total_sessions, total_sessions, len(checkpoint_images)))

        # ---- pre-built, FIXED-topology classifier head (Plan C §C1) -- built
        # ONCE here, over the full label space, and never touched again by
        # this driver. No per-cycle update_fc call anywhere below. ----
        self._network.update_fc(data_manager.nb_classes)

        self._cur_task = -1          # kept only because some _stream_* hooks
                                      # (e.g. InfLoRA's _init_lora_A) read it as
                                      # a generic "current slot index" counter;
                                      # NOT used here for any loss/head/eval
                                      # decision (leak-audit: see module docstring)
        self._stream_chunk = -1
        self._stream_init()

        # Private, write-only bookkeeping for offline analysis figures ONLY
        # (Plan C §C1: "the harness privately maps checkpoints to latent-task
        # progress... the model never sees it"). Both cumends arrays are read
        # here purely to LABEL results for later plotting/analysis -- neither
        # is ever read by any training, loss, head-growth, or eval-masking
        # decision above. task_image_cumends (cumulative IMAGE count per real
        # task) locates a stream POSITION within the task sequence;
        # task_class_cumends (cumulative CLASS count per real task) locates a
        # CLASS INDEX within the task sequence, used by _bounded_eval's
        # per-task accuracy breakdown below. FIXED 2026-07-25: these were
        # previously conflated into one array (task_sizes = class counts) and
        # used for BOTH purposes -- comparing an image-count position against
        # a class-count array silently saturated "nearest_latent_task" to the
        # max task index at every checkpoint (visible in tonight's earlier
        # smoke logs, which all reported the same "~4" for a 4-task run
        # regardless of actual progress). Analysis-only bug -- did not affect
        # any training, loss, or accuracy number reported tonight.
        task_image_cumends = np.cumsum(task_image_sizes)
        task_class_cumends = np.cumsum(task_class_sizes)

        # ---- metrics logging: runtime, persistent memory, peak VRAM, inference
        # FLOPs/latency (found MISSING entirely from this harness 2026-07-27 --
        # every prior bounded_memory run, interactive and the H200 production
        # grid alike, recorded accuracy only). ALWAYS ON here (unlike the oracle
        # path's "final_metrics" opt-in gate in trainer.py -- there is no other
        # track sharing this specific driver to protect from the overhead, and
        # this harness IS the production grid, so there is no case where these
        # metrics shouldn't be collected). Reuses utils/metrics_logger.py's
        # MetricsLogger completely unchanged -- same per-method
        # persistent_state()/_deployed_forward hooks the oracle path already
        # uses (confirmed present for all 5 round-2 methods: SeqLoRA/O-LoRA/
        # InfLoRA/TreeLoRA inherit LoRALearner's _deployed_forward, SketchLoRA
        # overrides its own) -- nothing here is SketchLoRA-specific or
        # method-specific in any way.
        _metrics_tag = "{}_{}_s{}".format(args["model_name"], args.get("prefix", "run"), seed0)
        mlog = MetricsLogger(os.path.join("run_logs", "final", args["model_name"]), _metrics_tag, args)
        self._bounded_metrics_path = mlog.out_path
        # Computational Efficiency (CE) metric (impl_plan_7.27.2026 Part 2) --
        # always on, same rationale as MetricsLogger above (no other track sharing
        # this driver to protect from the overhead, and per the plan's own cost
        # estimate this is logging, not compute: ~0 extra GPU-h beyond one profiler
        # measurement per run). N = cycle count under bounded-memory (sec 2.1),
        # so one ledger record per CYCLE, not per checkpoint.
        ce_ledger = OpsLedger(os.path.join("run_logs", "final", args["model_name"]), _metrics_tag)
        self._bounded_ce_ledger_path = ce_ledger.out_path
        _ce_step_macs = None   # (fwd, bwd) MACs, measured once on cycle 0, reused every cycle
        _ce_baseline_step_macs = None   # (fwd, bwd) MACs, R2 shared baseline, measured once alongside it

        # *** UNTESTED as of 2026-08-03 *** -- measured-CE region profiling
        # (docs/ce_profiling_implementation_plan.md, utils/ce_profiler.py). Additive
        # to the existing analytic-formula ledger fields above -- both are recorded
        # side by side per the plan's section 5/8 (the A/B comparison on identical
        # cycles is itself the validation criterion, not redundancy to clean up).
        # ce_profile_every=0 disables measured-region profiling entirely (formula
        # path is completely unaffected either way) -- a safety valve given nothing
        # in this file has touched a live run yet.
        _ce_profile_every = int(args.get("ce_profile_every", 25))
        ce_profile_controller = CEProfileController(
            self._device, profile_every=_ce_profile_every, enabled=_ce_profile_every > 0)

        results = []
        cum_images = 0
        next_ckpt_idx = 0
        cycle_idx = -1
        _prev_param_hash = None   # Round 2 §2.2 eval-routing identity check
        _prev_cycle_idx = None    # only compare hashes ACROSS a cycle boundary -- see below
        mlog.begin_task()   # starts timing the FIRST checkpoint-interval
        while cum_images < total_images:
            cycle_idx += 1
            c_start = cum_images
            c_end = min(total_images, c_start + cycle_images)
            _is_final_cycle = c_end >= total_images   # last cycle of the whole run -- force-profiled
            chunk_data = all_data[c_start:c_end]
            chunk_targets = all_targets[c_start:c_end]
            train_set = data_manager.get_dataset(
                [], source="train", mode="train", appendent=(chunk_data, chunk_targets))
            loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers)
            self._network.to(self._device)

            # Round-2 §1.1: cycle-local class mask, computed once per cycle from
            # this cycle's OWN raw targets only -- purely data-derived, no real
            # task/boundary information read. Classes absent from this cycle get
            # -inf added to their logit (zero softmax mass); classes present get
            # +0 (unchanged). Same additive-mask device/dtype as the network.
            cycle_classes = np.unique(chunk_targets)
            cycle_class_mask = torch.full(
                (data_manager.nb_classes,), float("-inf"), device=self._device)
            cycle_class_mask[torch.as_tensor(cycle_classes, device=self._device)] = 0.0

            # *** UNTESTED as of 2026-08-03 *** -- sets whether THIS cycle gets
            # region-profiled (docs/ce_profiling_implementation_plan.md sec 3.2).
            # MUST be called before _stream_begin_chunk below, NOT after -- a
            # real bug caught during Step 4 (InfLoRA) planning: _stream_begin_chunk
            # is where InfLoRA's _init_lora_A (a full extra forward pass + SVD,
            # genuinely substantial) actually runs, and ce_region() tags inside it
            # are no-ops unless a session is already active. Calling begin_cycle
            # after _stream_begin_chunk (the original ordering here) would have
            # made that entire cost permanently unmeasurable, on every cycle,
            # regardless of sampling cadence.
            ce_profile_controller.begin_cycle(cycle_idx, is_final=_is_final_cycle)

            # *** UNTESTED as of 2026-08-03 *** -- plan sec 4.3: _stream_begin_chunk
            # is the START-of-cycle boundary action (InfLoRA's _init_lora_A,
            # O-LoRA/TreeLoRA's add_task_slot, SketchLoRA's cycle-start reset) --
            # just as much a one-off per-cycle cost as _stream_end_chunk (the
            # END-of-cycle action, already wrapped further below), and the
            # EXISTING analytic formula already treats them as one combined
            # "boundary" charge (_ce_boundary_macs_this_cycle's
            # covariance_hooks_base_forward uses "2 * chunk_images * ..." to cover
            # BOTH _init_lora_A's and _update_dualgpm's extra forward passes
            # together). Tracked under its own controller kind
            # ("boundary_begin") rather than reusing "boundary" (which
            # _stream_end_chunk uses below) because CEProfileController.commit()
            # overwrites its held value per kind -- using two kinds and merging
            # their dicts at record_unit() below is simpler and safer than
            # teaching the controller an accumulate-into-existing-dict mode.
            with ce_profile_controller.session("boundary_begin") as _boundary_begin_sess:
                self._stream_begin_chunk(loader)
            ce_profile_controller.commit(_boundary_begin_sess, "boundary_begin", scale=1.0)

            self._bounded_new_optimizer()   # Round-2 §1.2: head weight_decay=0, uniform
            optimizer, scheduler = self._stream_optim, self._stream_sched

            # *** UNTESTED as of 2026-08-03 *** -- CHANGED (user-directed, 2026-08-03,
            # in response to the CE-smoke-test request): re-measure Ops_fb/baseline on
            # EVERY profiled cycle, not just once at cycle 0. Originally this ran once
            # ever and was reused for the whole run -- fine for O-LoRA/InfLoRA/TreeLoRA,
            # whose folded merge=True forward is genuinely architecturally constant, but
            # WRONG for SketchLoRA: cycle 0's sketch slot is still zero (unpopulated,
            # pre-first-fold), so a cycle-0-only measurement would freeze
            # merged_forward_excess_per_step (the R2 quantity that isolates the
            # (B_hat @ A_hat) @ x sketch-inclusion cost) at ~0 for the ENTIRE run,
            # exactly hiding the one number this whole exercise wants to see grow with
            # r_hat. Gating on ce_profile_controller.profiling_this_cycle reuses the
            # SAME sampling cadence as every other measured region (still held between
            # samples, same as everything else) -- `or _ce_step_macs is None` guarantees
            # at least one measurement happens even if ce_profile_every=0 disables
            # sampling entirely.
            if _ce_step_macs is None or ce_profile_controller.profiling_this_cycle:
                # Ops_fb measurement (impl_plan_7.27.2026 sec 2.2: "measured, not
                # assumed"), AFTER _stream_begin_chunk so trainability/slot routing
                # match real training exactly. zero_grad() after: this is a throwaway
                # measurement, must not leak into the real optimizer's first step.
                #
                # R2 (plan sec 2): ALSO measures a shared, method-independent baseline
                # (single slot, merge=False -- the SeqLoRA-equivalent configuration)
                # alongside the method's own actual routing, so a method's own extra
                # forward cost (O-LoRA/InfLoRA/TreeLoRA's frozen_delta matmul,
                # SketchLoRA's sketch-slot matmul) is charged rather than cancelling
                # between numerator and denominator.
                _probe_inputs, _probe_targets = next(iter(loader))[1:]
                _probe_inputs = _probe_inputs.to(self._device)
                _probe_targets = _probe_targets.to(self._device)
                _slot, _merge = self._stream_slot(), self._stream_train_merge()

                def _loss_fn(logits):
                    return F.cross_entropy(logits + cycle_class_mask, _probe_targets)

                _baseline_fwd, _baseline_bwd, _actual_fwd, _actual_bwd = measure_baseline_and_actual(
                    self._network, _probe_inputs, _probe_targets, _loss_fn, _slot, _merge, self._device)
                _ce_baseline_step_macs = (_baseline_fwd, _baseline_bwd)
                _ce_step_macs = (_actual_fwd, _actual_bwd)
                self._network.zero_grad()

            # Measured-CE region profiling, epoch 0 only of a profiled cycle
            # (docs/ce_step_boundary_isolation_plan.md sec 1a/2/7/8, REWORKED
            # 2026-08-05 -- was: wrap the ENTIRE _bounded_train_epoch call under
            # torch.profiler, tracing every op of every batch's full ViT
            # forward+backward+optimizer.step() just to extract a few small
            # tagged matmuls. NarrowAuxAccumulator instead lets the method wrap
            # ONLY its own isolated aux call, every step of epoch 0 -- the
            # profiler now only ever instruments the small aux computation, not
            # the (much larger, shared-with-every-method, already-measured-
            # separately-via-R2-above) base training step. step_acc accumulates
            # the RAW (undivided) sum across all of epoch 0's steps;
            # split_by_recurrence below separates genuinely-per-step tags from
            # per-epoch ones (e.g. TreeLoRA's tree_search_first_call, which
            # fires once per epoch not once per step -- see that module) BEFORE
            # applying the per-step average scaling, so a per-epoch cost is
            # never diluted-then-wrongly-rescaled the way it would be if lumped
            # into the per-step bucket blindly.
            step_acc = NarrowAuxAccumulator(self._device, enabled=ce_profile_controller.profiling_this_cycle)
            for _ep in range(epochs):
                if _ep == 0:
                    self._bounded_train_epoch(loader, optimizer, scheduler, cycle_class_mask,
                                              step_acc=step_acc)
                else:
                    self._bounded_train_epoch(loader, optimizer, scheduler, cycle_class_mask)
            _measured_step, _measured_per_epoch = split_by_recurrence(
                step_acc.totals(), step_scale=1.0 / max(len(loader), 1))

            # *** UNTESTED as of 2026-08-03 *** -- R7 fix (plan sec 6 item 1): the
            # formula-based aux cost must be snapshotted BEFORE _stream_end_chunk
            # runs, not after. The ORIGINAL code called self._ce_aux_macs_per_step()
            # inside record_unit() below, i.e. AFTER _stream_end_chunk had already
            # fired -- for SketchLoRA specifically, _stream_end_chunk's _compress()
            # call changes r_hat (the sketch's rank) via a fold, so the aux formula
            # was reading the POST-fold r_hat, not the value that was actually in
            # force while this cycle's training steps ran. Moving the read here
            # fixes that for every method uniformly (a no-op change for methods
            # whose aux state doesn't change at the boundary).
            _ce_aux_macs_formula = self._ce_aux_macs_per_step()

            with ce_profile_controller.session("boundary_end") as _boundary_end_sess:
                self._stream_end_chunk(loader)
            ce_profile_controller.commit(_boundary_end_sess, "boundary_end", scale=1.0)

            # *** UNTESTED as of 2026-08-03 *** -- merge the begin- and
            # end-of-cycle boundary region dicts into one (plain dict-union;
            # region labels are namespaced per method/call-site, e.g.
            # "inflora/init_lora_A_forward" vs "inflora/update_dualgpm_forward",
            # so the two dicts are not expected to share keys in practice -- if
            # they ever did, this union keeps boundary_end's value, silently, for
            # that key only. Matches the pre-existing analytic formula's own
            # convention of charging begin- and end-of-cycle one-off costs into
            # the SAME "boundary_macs" category (e.g. InfLoRA's
            # covariance_hooks_base_forward already combines both extra passes).
            _measured_boundary = {**ce_profile_controller.current("boundary_begin"),
                                  **ce_profile_controller.current("boundary_end")}

            # *** UNTESTED as of 2026-08-03 *** -- added for the CE smoke test
            # (user request: "data at each task boundary, so we can see how
            # each method's compute is beginning to scale with task count").
            # Same computation _bounded_eval's own "_nearest_latent_task" field
            # already uses, just applied every cycle instead of only at
            # accuracy checkpoints -- write-only telemetry, per this module's
            # own leak-audit convention (see the module docstring).
            _nearest_latent_task = int(np.searchsorted(task_image_cumends, c_end))

            ce_ledger.record_unit(
                unit_idx=cycle_idx, steps_per_epoch=len(loader), n_epochs=epochs,
                step_macs_fwd=_ce_step_macs[0], step_macs_bwd=_ce_step_macs[1],
                aux_macs_per_step=_ce_aux_macs_formula,
                boundary_macs=self._ce_boundary_macs_this_cycle(
                    len(chunk_data), macs_per_image_fwd=_ce_step_macs[0] / self.batch_size),
                measured_step_regions=_measured_step,
                measured_per_epoch_regions=_measured_per_epoch,
                measured_boundary_regions=_measured_boundary,
                baseline_step_macs_fwd=_ce_baseline_step_macs[0],
                baseline_step_macs_bwd=_ce_baseline_step_macs[1],
                nearest_latent_task=_nearest_latent_task,
                profile_provenance={
                    "step": {"profiled": bool(ce_profile_controller.profiling_this_cycle)},
                    "boundary_begin": ce_profile_controller.provenance("boundary_begin"),
                    "boundary_end": ce_profile_controller.provenance("boundary_end"),
                })

            cum_images = c_end
            # CHECKPOINT-interval granularity, not per-cycle: a checkpoint spans
            # ~cycles_per_checkpoint (often 5-10) training cycles, and
            # mlog.begin_task()/mark_train_done() bracket exactly that whole span --
            # so train_seconds below is the REAL accumulated training time since the
            # last checkpoint, not a fraction of it. persistent_state()/peak-VRAM/
            # disk-write only happen at checkpoint frequency (matching
            # _bounded_checkpoint_write's own existing cadence exactly, not more
            # often) -- doing this every raw cycle instead would multiply disk I/O
            # and the persistent_state() python-loop 5-10x for zero measurement
            # benefit, since nothing about persistent state needs finer-than-
            # checkpoint resolution to be meaningful.
            checkpoint_due = (next_ckpt_idx < len(checkpoint_images)
                              and cum_images >= checkpoint_images[next_ckpt_idx])
            if checkpoint_due:
                mlog.mark_train_done()   # ends TRAIN time for the whole interval just finished
            cnn_accy = None
            while next_ckpt_idx < len(checkpoint_images) and cum_images >= checkpoint_images[next_ckpt_idx]:
                acc, acc5, hi_total, per_task_acc, per_task_acc5 = self._bounded_eval(
                    all_data, all_targets, cum_images, data_manager, task_class_cumends)
                nearest_task = int(np.searchsorted(task_image_cumends, cum_images))
                param_hash = self._bounded_param_hash()
                # FIXED (2026-07-28): the real invariant is "params changed ACROSS a
                # cycle boundary" (real training happened, so a stale/frozen eval state
                # would be a bug) -- NOT "params changed between any two consecutive
                # checkpoints." When checkpoint density exceeds cycle count (e.g.
                # ImageNet-R at 200MB budget: 18 cycles vs the dataset's 20 requested
                # checkpoint fractions), multiple checkpoints legitimately land on the
                # SAME cycle with no training in between -- an unchanged hash there is
                # CORRECT, not a bug (confirmed via a crash on exactly this: "cycle 3 ->
                # 3" in the old message format, which already anticipated printing the
                # same cycle twice without ever guarding against it).
                if _prev_param_hash is not None and cycle_idx != _prev_cycle_idx:
                    assert param_hash != _prev_param_hash, (
                        "Round 2 §2.2 eval-routing identity check FAILED: parameter-state "
                        "hash did not change across a cycle boundary (cycle {} -> {}) -- "
                        "eval may be reading a stale/frozen state instead of the "
                        "just-trained one.".format(_prev_cycle_idx, cycle_idx))
                _prev_param_hash = param_hash
                _prev_cycle_idx = cycle_idx
                cnn_accy = {"top1": acc}   # feeds mlog.record_task below, once per checkpoint
                logging.info(
                    "[bounded_mem eval] volume {:.2f} | cycle {} | classes_seen {} | "
                    "CIL top1 {:.2f} | top5 {:.2f} | (nearest latent task ~{}) | "
                    "param_hash {} | per-task {}".format(
                        checkpoint_images[next_ckpt_idx] / total_images, cycle_idx,
                        hi_total, acc, acc5, nearest_task, param_hash, per_task_acc))
                results.append({
                    "completed_frac": round(checkpoint_images[next_ckpt_idx] / total_images, 4),
                    "cum_images": checkpoint_images[next_ckpt_idx],
                    "cil": acc,
                    "cil_top5": acc5,
                    "classes_seen": hi_total,
                    "cycle": cycle_idx,
                    "_nearest_latent_task": nearest_task,   # analysis-only, see docstring
                    "_param_hash": param_hash,              # Round 2 §2.2 eval-routing identity
                    "per_task_acc": per_task_acc,           # forgetting curve: [acc on task 0's
                                                              # classes, task 1's, ...] at THIS
                                                              # checkpoint, analysis-only
                    "per_task_acc_top5": per_task_acc5,
                })
                _bounded_checkpoint_write(args, results, total_sessions, cycle_images,
                                          budget_mb, total_images)
                next_ckpt_idx += 1

            if checkpoint_due:
                mlog.record_task(self, cycle_idx, cnn_accy, None)
                mlog.begin_task()   # start timing the NEXT checkpoint-interval immediately

        # ---- final inference-cost measurement (FLOPs/latency) + finalize ----
        # Measured once, at the end, over the full final class range -- same
        # convention as the oracle path's record_inference_cost call. cnn_matrix/
        # til_matrix are intentionally None: bounded_memory has no real per-task
        # boundary matrix to compute FAA/AIA/forgetting/BWT from (this harness is
        # explicitly boundary-free -- see module docstring), so finalize() just
        # marks the metrics JSON "done" without attempting a CL-summary that
        # doesn't have a well-defined meaning here (mirrors how TIL is already
        # "not computed -- meaningless in the memory-increment setup").
        final_test_set = data_manager.get_dataset(
            np.arange(0, data_manager.nb_classes), source="test", mode="test")
        final_test_loader = DataLoader(final_test_set, batch_size=64, shuffle=False,
                                       num_workers=num_workers)
        mlog.record_inference_cost(self, final_test_loader)
        mlog.finalize(None, None, cycle_idx)

        # CE metric summary (impl_plan_7.27.2026 sec 2.1): eps = self.epochs, the
        # shared epoch budget (E=20 in this campaign) -- computed OFFLINE from the
        # just-written ledger, logged here for quick reference; the ledger itself
        # (ce_ledger.out_path) is the artifact anything downstream should read from.
        #
        # *** UNTESTED as of 2026-08-03 *** -- switched to compute_ce_report
        # (docs/ce_profiling_implementation_plan.md), which reports the ORIGINAL
        # formula-based CE (ce_formula, byte-identical to the old compute_ce(...)
        # call this replaces) SIDE BY SIDE with the new measured-region CE
        # (ce_measured) and both under the R2 shared-baseline numerator
        # (*_baseline_numerator) -- plus n_actually_profiled, so it's never
        # ambiguous how many of this run's cycles the measured numbers are really
        # based on versus held from a prior sample. Headline logged value is
        # ce_best (CHANGED 2026-08-03, matching trainer.py's oracle-path
        # logging: the measured path is the one this whole exercise exists to
        # trust, and every bounded_memory run now always populates the measured
        # fields, so ce_best resolves to ce_measured_baseline_numerator here in
        # practice) -- the full report (incl. ce_formula) stays in the log line
        # for anyone who wants the old number for comparison.
        from utils.ops_ledger import compute_ce_report
        ce_report = compute_ce_report(ce_ledger.records, eps=self.epochs)
        logging.info("[CE metric] {} = {} (source={}) (eps={}, N={} cycles) | full report: {}".format(
            args["model_name"], ce_report["ce_best"] if ce_report else None,
            ce_report["ce_best_source"] if ce_report else None,
            self.epochs, len(ce_ledger.records), ce_report))

        self._bounded_results = results
        return results


def _bounded_checkpoint_write(args, results, total_sessions, cycle_images, budget_mb, total_images):
    """Incremental write, same rationale as stream_mixin.py's
    _stream_checkpoint_write (models/stream_mixin.py): a hard kill mid-run
    (Plan C's own 12h wall-clock budget, or the campaign's) must still leave a
    real, structured, partial result on disk, not just log text."""
    seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed")
    out = "run_logs/boundedmem_{}_{}_s{}.json".format(args["model_name"], args.get("prefix", "run"), seed)
    os.makedirs("run_logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "args_subset": {k: args.get(k) for k in
                ("model_name", "dataset", "init_cls", "increment", "bm_budget_mb",
                 "n_lora_blocks", "init_lr", "svd_energy_target", "lamda_1", "lamb", "lame",
                 "sketchlora_admission", "sketchlora_rank_cap", "sketchlora_lora_wd")},
            "budget_mb": budget_mb, "cycle_images": cycle_images, "total_sessions": total_sessions,
            "total_images": total_images, "results": results, "partial": True,
        }, f, indent=2)
