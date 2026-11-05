"""Unique-image memory-budget streaming for the LoRA/prompt CL methods.

The standard LAMDA-PILOT loop (trainer.py) calls ``incremental_train`` once per
class-task: head grows, N epochs train, then the method's boundary action
(SeqLoRA optimizer reset / SketchLoRA compress / O-LoRA new slot / InfLoRA new
slot + DualGPM / ...) fires AT the task end -- coupling the model's adapter
boundaries to the ground-truth class-task boundaries.

This mixin adds a parallel streaming codepath (``boundary_mode: sample``) that
genuinely simulates a memory-constrained data stream the model is unaware of:

  * The full dataset is treated as ONE fixed, non-repeating sequence of unique
    images: each real task contributes its own images in a single fixed random
    order (mixing that task's own classes together -- an "arrival order," not
    the ordinary per-epoch minibatch shuffle), concatenated task-by-task.
  * That sequence is partitioned into fixed-size chunks of exactly
    ``stream_chunk_samples`` images (2026-10-29: a direct raw sample count --
    the project's standing param for this now, set it to whatever chunk size
    "r" you want tested; ``stream_budget_mb`` still works too, converted via
    the same 224x224x3 bytes/image accounting as utils/budget_stream.py, kept
    only for existing configs that already use it -- new configs should use
    stream_chunk_samples directly, it's the literal quantity the mechanism
    actually operates on, not a size estimate that has to be reverse-engineered
    back into an image count). A chunk that only partially overlaps a task
    carries the task's remaining images forward into the very next chunk -- no
    image is ever shown twice, and none are skipped.
  * A chunk trains for the method's own ``epochs`` epochs (repeated passes over
    that FIXED chunk are normal and don't consume additional budget -- the
    budget counts distinct images once, not per-epoch exposure).
  * The classifier head and CE-loss range are derived directly from whichever
    classes are actually present in the current chunk's images (which can
    include a next task's classes the moment any of its images land in the
    current chunk) -- NOT from a separate, independently-clocked real-task
    loop. This is the one deliberately-unhidden signal: task/class identity is
    still visible via the loss mask and head size, exactly as in ordinary
    task-incremental learning; only each method's OWN adapter bookkeeping
    (fold/new slot/compress/etc, fired once per chunk) is decoupled from real
    task boundaries.
  * A real task becomes eligible for its own CIL checkpoint the moment a
    chunk's cumulative image count first reaches that task's own cumulative
    image count -- multiple tasks can complete within one large chunk, or one
    task can span many small chunks, with no special-casing either way.

The routed slot (``_stream_slot``) is decoupled from the class-task index:
SeqLoRA pins slot 0; SketchLoRA uses sketch(0)/residual(1); O-LoRA/InfLoRA/etc
advance one slot per CHUNK. Each method overrides the small chunk hooks and
reuses its own math primitives (compress, orthogonality, DualGPM, warm-start,
...) -- none of those hooks change under this redesign; only stream_run()'s own
data-preparation and outer-loop logic does.
"""

import json
import logging
import os
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

# 2026-11-02: 8->4 (see models/lora.py's identical earlier change for the
# original rationale). Combined with persistent_workers=True on every loader
# that's iterated more than once over its own lifetime (the per-chunk
# training loaders below, reused across tuned_epoch epochs) -- without it,
# PyTorch respawns the ENTIRE worker pool at the start of every epoch
# iteration, not just once per chunk; on Windows (spawn-based multiprocessing,
# no fork()) this is genuinely slow (whole-module reimport per worker) and
# was confirmed to dominate stream_run()'s wall-clock time on this machine
# (8.5 min between chunk boundaries, 4-5x more than the ~2 min the actual
# forward/backward compute needs for a 500-image chunk) -- see
# feedback_windows_dataloader_respawn_bottleneck memory. Likely far less
# severe on the Linux cluster (fork() is a cheap copy-on-write clone, not a
# full reimport) -- the cluster's own earlier fix (raising --cpus-per-task)
# addressed a different problem, core contention among live workers, not
# this respawn cost.
num_workers = 4


def _stream_checkpoint_write(args, results, tag="stream", partial=True):
    """Write the results-so-far next to where trainer.py::_run_stream will write
    the final copy (same path convention: run_logs/{tag}_{model_name}_{prefix}_
    s{seed}.json). Called after EVERY per-task checkpoint inside stream_run's own
    loop (not just once at the very end) so a hard kill mid-run -- e.g. a wall-
    clock budget cutoff -- still leaves a real, structured accuracy record on
    disk instead of losing everything back to whatever's only in the log text.
    The final call from trainer.py (partial=False) overwrites this with the
    complete run, so there's no dual-source-of-truth conflict on clean exit."""
    seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed")
    out = "run_logs/{}_{}_{}_s{}.json".format(tag, args["model_name"], args.get("prefix", "run"), seed)
    os.makedirs("run_logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args_subset": {k: args.get(k) for k in
                    ("model_name", "dataset", "init_cls", "increment", "stream_chunk_samples",
                     "stream_budget_mb", "n_lora_blocks", "init_lr", "svd_energy_target",
                     "lamda_1", "lamb", "lame", "plateau_loss_window_length",
                     "plateau_ema_beta", "plateau_variance_ratio",
                     "plateau_max_chunk_samples")},
                    "results": results, "partial": partial}, f, indent=2)

# uint8 model-input accounting: 224x224x3, the same "memory budget" convention
# used by utils/budget_stream.py -- kept identical so MB values mean the same
# thing across both streaming designs.
BYTES_PER_IMAGE = 224 * 224 * 3


def _build_stream_order(data_manager, seed0, n_run):
    """The fixed, seeded, per-task-permuted, task-concatenated flat image/
    label stream shared by every boundary-agnostic streaming mode (factored
    out of stream_run() 2026-10-30 so plateau_run() -- a different CHUNKING
    policy over the exact same stream -- operates on identical data/order,
    not a re-derived approximation of it). Every real task's own images get
    ONE fixed random permutation (mixing that task's classes together, this
    task's own "arrival order"), then all tasks' permuted streams are
    concatenated in task order. Returns (task_ranges, all_data, all_targets,
    task_img_cumends) -- task_img_cumends[t] is the cumulative UNIQUE image
    count through the end of real task t, used by every mode to decide when
    a real task becomes eligible for its own CIL checkpoint."""
    task_ranges = []
    data_parts, targets_parts = [], []
    task_img_cumends = []
    running = 0
    known = 0
    for t in range(n_run):
        task_size = data_manager.get_task_size(t)
        lo, hi = known, known + task_size
        task_ranges.append((lo, hi))
        task_data, task_targets, _ = data_manager.get_dataset(
            np.arange(lo, hi), source="train", mode="train", ret_data=True)
        perm = np.random.RandomState(seed0 * 9973 + t).permutation(len(task_data))
        data_parts.append(task_data[perm])
        targets_parts.append(task_targets[perm])
        running += len(task_data)
        task_img_cumends.append(running)
        known = hi
    all_data = np.concatenate(data_parts)
    all_targets = np.concatenate(targets_parts)
    return task_ranges, all_data, all_targets, task_img_cumends


class StreamMixin:
    # ---- per-method hooks (defaults = SeqLoRA-style single slot) ----
    def _stream_init(self):
        """Per-method state set up once before the first chunk."""
        pass

    def _stream_slot(self):
        """Backbone slot index to train/route during the current chunk."""
        return 0

    def _stream_train_merge(self):
        return bool(getattr(self, "train_merge", False))

    def _stream_begin_chunk(self, loader):
        """Start a new adapter chunk: (advance slot,) set trainability, fresh optimizer."""
        self._network.freeze_to_task(self._stream_slot(),
                                     train_a=getattr(self, "_stream_train_a", True))
        for p in self._network.fc.parameters():
            p.requires_grad = True
        self._stream_new_optimizer()

    def _stream_end_chunk(self, loader):
        """Adapter boundary action (fold/snapshot/subspace). Default: nothing
        (SeqLoRA's single adapter drifts; the 'reset' is just the next optimizer)."""
        pass

    def _stream_end_task(self, ct):
        """Fired once per REAL task, right after that task's own epochs finish
        (unlike _stream_end_chunk, which fires on the decoupled adapter clock).
        For methods whose bookkeeping is genuinely task-scoped rather than
        chunk-scoped (e.g. HiDeLoRA's per-class centroid stats + TAP head
        recalibration -- computing those per-CHUNK would reintroduce the same
        duplicate-class/no-faithful-merge problem that excluded EASE/TUNA/
        CL-LoRA from memory-constrained training entirely). Default: nothing."""
        pass

    def _stream_extra_loss(self, lo, hi):
        """Optional added loss (O-LoRA orthogonality). Default: 0."""
        return 0.0

    # -- Computational Efficiency (CE) metric hooks (impl_plan_7.27.2026 Part 2,
    # utils/ops_ledger.py + utils/ce_formulas.py). Both default to "no auxiliary
    # cost" -- SeqLoRA needs no override (its CE=1.0 sanity anchor IS this
    # default); O-LoRA/InfLoRA/TreeLoRA override one or the other per their own
    # actual mechanism (see models/olora.py, inflora.py, treelora.py). ----
    def _ce_aux_macs_per_step(self):
        """Extra MACs charged EVERY training step, on top of the measured
        fwd+bwd (e.g. O-LoRA's per-step orthogonality penalty, TreeLoRA's
        per-step regularizer). Default: 0."""
        return 0.0

    def _ce_boundary_macs_this_cycle(self, chunk_images, macs_per_image_fwd=0.0):
        """One-off MACs charged once per CYCLE regardless of step count (e.g.
        InfLoRA's two extra full passes over the chunk for cur_matrix
        accumulation). Returns an itemized dict; default: none.

        macs_per_image_fwd: the profiler-measured forward-only MACs for ONE
        image (bounded_memory_mixin.py passes step_macs_fwd/batch_size, the
        same one-time profiler measurement already used for Ops_fb). Needed
        by any method whose boundary cost includes running full extra
        forward passes over chunk-sized data (currently only InfLoRA's
        DualGPM covariance-accumulation passes) -- the BASE cost of running
        the pass at all, not just whatever incremental bookkeeping rides
        along on top of it. Default 0.0 is a no-op for every other method
        (O-LoRA, TreeLoRA, SketchLoRA, SeqLoRA), whose boundary costs don't
        involve extra full forward passes over the data."""
        return {}

    def _stream_cil_forward(self, inputs):
        """Deployed (CIL) forward: route current slot with the method's merge flag."""
        return self._network(inputs, task=self._stream_slot(), merge=self._stream_train_merge())

    # ---- shared machinery ----
    def _stream_new_optimizer(self):
        params = [p for p in self._network.parameters() if p.requires_grad]
        self._stream_optim = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)
        # lr_anneal=false -> constant init_lr (no per-chunk cosine); de-confounds the
        # sample-boundary runs by removing the schedule that would misalign with the chunk clock.
        # T_max: previously self._stream_boundary_every (a fixed epoch-count under the
        # old epoch-clock design). Under the sample-count redesign, chunks have
        # VARIABLE epoch length (a chunk's real duration depends on which task's data
        # is being processed when the sample threshold fires), so there's no single
        # exact "epochs per chunk" value anymore. Using self.epochs (the standard
        # per-task epoch count) as a reasonable proxy -- AUTONOMOUS APPROXIMATION,
        # flagged in BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md for review.
        self._stream_sched = optim.lr_scheduler.CosineAnnealingLR(
            self._stream_optim, T_max=self.epochs, eta_min=self.min_lr) \
            if getattr(self, "lr_anneal", True) else None

    def _stream_train_epoch(self, loader, lo, hi):
        # TEMPORARY DIAGNOSTIC (2026-11-03): counting actual batches/images
        # iterated vs. len(loader.dataset) to check whether plateau_run()'s
        # retroactive epoch is silently delivering fewer batches than its
        # own segment size implies (suspected DataLoader persistent_workers
        # leak across fires -- see conversation). Remove once resolved.
        import time as _time
        _t0 = _time.time()
        _n_batches = 0
        _n_images = 0
        _expected_images = len(loader.dataset)
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=slot, merge=merge)["logits"]
            local_logits = logits[:, lo:hi]
            loss = F.cross_entropy(local_logits, targets - lo)
            extra = self._stream_extra_loss(lo, hi)
            if not (isinstance(extra, float) and extra == 0.0):
                loss = loss + extra
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()
            _n_batches += 1
            _n_images += inputs.size(0)
        logging.info("[DIAG _stream_train_epoch] expected_images={} actual_images={} "
                     "batches={} elapsed={:.3f}s ({:.4f}s/image) MATCH={}".format(
                         _expected_images, _n_images, _n_batches, _time.time() - _t0,
                         (_time.time() - _t0) / max(_n_images, 1), _n_images == _expected_images))
        if self._stream_sched is not None:
            self._stream_sched.step()

    # ---- the streaming driver ----
    def stream_run(self, data_manager, args):
        """Unique-image memory-budget redesign (2026-07-19, supersedes the
        epoch-repeat sample-count clock). The earlier design counted CUMULATIVE
        EPOCH-REPEATED samples toward the budget -- training 10 epochs over the
        same 1000 images counted as 10000 "samples," which made the clock a
        proxy for compute/time elapsed, not for how much DISTINCT data the model
        has access to. That caused a real, discovered problem: a dataset whose
        per-epoch task volume alone exceeds the byte threshold (e.g. Food101)
        made 250MB and 500MB produce IDENTICAL fold schedules, since both
        thresholds were crossed within a single epoch either way.

        This version instead treats the budget as literally how many UNIQUE
        images the model may hold at once -- a genuine bounded-memory /
        streaming-buffer simulation:
          * Every real task's own images get ONE fixed random permutation
            (mixing that task's classes together, computed once, seeded for
            reproducibility) -- this is the task's own "arrival order" in the
            stream, distinct from ordinary per-EPOCH minibatch shuffling (which
            still happens normally via DataLoader(shuffle=True) below).
          * All tasks' permuted image streams are concatenated in task order,
            then sliced into fixed-size chunks of exactly ``budget_images``
            images each (the last chunk may be smaller). No image ever appears
            in two chunks; a chunk that only partially overlaps a task carries
            the OTHER half forward into the very next chunk, never repeating
            and never re-shuffling.
          * A chunk trains for the method's own ``epochs`` (repeated passes
            over that chunk's own fixed image set -- unlike before, repeat
            epochs do NOT consume additional budget, since the budget now
            counts distinct images once, not per-epoch exposure).
          * The classifier head grows to cover whatever classes are actually
            present in the current chunk (``max(chunk labels) + 1``) BEFORE
            that chunk trains -- this can include a next task's classes the
            moment any of its images fall inside the current chunk, exactly
            as dictated by which images the chunk happens to contain. The CE
            loss range is likewise derived directly from the chunk's own
            label content (reusing the same [min,max+1) pattern already
            validated for utils/budget_stream.py's identical multi-class-
            chunk problem), NOT the task-boundary-based [known,total) range.
          * A real task is "completed" (eligible for its own CIL checkpoint)
            the moment a chunk's cumulative image count reaches or passes that
            task's own cumulative image count -- computed directly from the
            precomputed schedule, so multiple tasks can complete within a
            single large chunk, or a single task can span many small chunks,
            with no separate dedupe logic needed (each chunk advances the
            "highest completed task" pointer monotonically, at most once per
            task, by construction).

        Per-method hooks (_stream_begin_chunk/_stream_end_chunk/_stream_
        train_epoch/_stream_cil_forward/_stream_end_task) are COMPLETELY
        UNCHANGED by this redesign -- they already take the chunk's loader and
        an explicit (lo, hi) CE range as arguments/state, agnostic to how chunk
        boundaries were computed. Only this method's own data-preparation and
        outer-loop logic changes.
        """
        self.data_manager = data_manager
        epochs = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[stream] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))
        # 2026-10-29: stream_chunk_samples (a direct raw image/sample count,
        # "r") is now the preferred way to set the chunk size -- the literal
        # quantity stream_run() actually slices by, not an MB estimate that
        # gets converted back into one. stream_budget_mb is still accepted
        # (for existing pre-2026-10-29 configs, e.g. exps/omni1k_streamcompare/
        # *.json) and converted the same way as before if stream_chunk_samples
        # isn't set. Exactly one of the two must be present -- no silent
        # default, same contract as the old stream_budget_mb-only version.
        chunk_samples_arg = args.get("stream_chunk_samples")
        stream_budget_mb_arg = args.get("stream_budget_mb")
        assert (chunk_samples_arg is not None) != (stream_budget_mb_arg is not None), \
            "boundary_mode='sample' needs exactly one of stream_chunk_samples (preferred, " \
            "a direct image count) or stream_budget_mb (legacy, converted via the same " \
            "224x224x3 bytes/image accounting as utils/budget_stream.py) -- got " \
            "stream_chunk_samples={}, stream_budget_mb={}".format(chunk_samples_arg, stream_budget_mb_arg)
        if chunk_samples_arg is not None:
            budget_images = max(1, int(chunk_samples_arg))
            stream_budget_mb = None
        else:
            stream_budget_mb = float(stream_budget_mb_arg)
            budget_images = max(1, round(stream_budget_mb * 1024 * 1024 / BYTES_PER_IMAGE))
        # Opt-in (default off, preserves existing per-completed-task eval behavior for
        # any caller relying on the full accuracy-vs-chunk curve): skip the CIL eval that
        # would otherwise fire every time a chunk completes one or more real tasks, and
        # only evaluate once at the very end, over however many tasks actually completed.
        # Does NOT change _stream_end_task's per-task bookkeeping (bank/slot growth etc.)
        # -- only the accuracy-computation eval call is deferred.
        eval_only_final = bool(args.get("stream_eval_final_only", False))

        seed_arg = args.get("seed", 1993)
        seed0 = seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg

        # ---- precompute the fixed, non-repeating unique-image stream ----
        self._task_ranges, all_data, all_targets, task_img_cumends = \
            _build_stream_order(data_manager, seed0, _n_run)
        total_images = len(all_targets)

        chunk_bounds = list(range(0, total_images, budget_images)) + [total_images]
        chunks = [(chunk_bounds[i], chunk_bounds[i + 1]) for i in range(len(chunk_bounds) - 1)
                  if chunk_bounds[i + 1] > chunk_bounds[i]]
        budget_desc = "{} images (direct sample count)".format(budget_images) if stream_budget_mb is None \
            else "{} images ({}MB)".format(budget_images, stream_budget_mb)
        logging.info("[stream] unique-image chunk size {}; {} total images, "
                     "{} tasks -> {} chunks".format(budget_desc, total_images, _n_run, len(chunks)))

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        last_task_ended = -1
        results = []

        for c_start, c_end in chunks:
            chunk_data = all_data[c_start:c_end]
            chunk_targets = all_targets[c_start:c_end]
            lo, hi = int(chunk_targets.min()), int(chunk_targets.max()) + 1
            if hi > self._total_classes:
                self._total_classes = hi
                self._network.update_fc(self._total_classes)
            train_set = data_manager.get_dataset(
                [], source="train", mode="train", appendent=(chunk_data, chunk_targets))
            # persistent_workers=True: this loader is reused across `epochs` epoch
            # iterations below (_stream_train_epoch is called `epochs` times on the
            # SAME loader object) -- without it, the worker pool respawns every
            # epoch, not just once per chunk. See num_workers' own comment above.
            loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True)
            self._network.to(self._device)

            self._stream_begin_chunk(loader)
            for _ep in range(epochs):
                self._stream_train_epoch(loader, lo, hi)
            self._stream_end_chunk(loader)

            newly_completed = []
            t = last_task_ended + 1
            while t < _n_run and task_img_cumends[t] <= c_end:
                newly_completed.append(t)
                t += 1
            for t in newly_completed:
                self._stream_task_to_chunk[t] = self._stream_slot()
                self._stream_end_task(t)
            if newly_completed:
                last_task_ended = newly_completed[-1]
                if not eval_only_final:
                    results.append(self._stream_eval(last_task_ended + 1))
                    _stream_checkpoint_write(args, results, tag="stream", partial=True)

        if eval_only_final and last_task_ended >= 0:
            results.append(self._stream_eval(last_task_ended + 1))
            _stream_checkpoint_write(args, results, tag="stream", partial=True)

        self._stream_results = results
        return results

    # ---- loss-plateau adapter boundaries (2026-10-30, ported from Online-LoRA) ----
    def plateau_run(self, data_manager, args):
        """SketchLoRA-only boundary mode: instead of chunking the SAME fixed
        stream (_build_stream_order, shared with stream_run() above) into
        fixed-size pieces, the chunk boundary itself is decided by a loss-
        plateau detector (utils/online_lora_plateau.py::LossPlateauDetector,
        ported from Online_LoRA_Ref/Si-blurry/online_lora.py's own
        distribution-shift detection -- see that module's docstring for the
        mechanism itself; only the detection state machine is ported, not
        Online-LoRA's MAS regularizer or hard-buffer replay).

        Per-segment lifecycle (a "segment" = the variable-length span of the
        stream between two fires):
          1. Online phase: ONE single first-exposure pass, strict stream
             order (no shuffle -- this IS the arrival order), feeding each
             step's plain CE loss (pre-_stream_extra_loss, same reasoning as
             the reference excluding its own regularizer from the window)
             into the detector. This is what makes the detector's signal
             meaningful -- loss on already-seen data trends to zero
             regardless of whether a real shift occurred, which would make
             a plateau condition fire almost immediately on every segment,
             uninformatively; the detector has to see genuinely fresh data.
          2. The moment the detector fires, the segment's extent is known.
             If self.epochs > 1, retroactively run self.epochs - 1 MORE
             passes over exactly that now-closed segment (ordinary shuffled
             multi-epoch training via the existing _stream_train_epoch) --
             this is what actually trains the adapter to convergence before
             it's folded; the single online pass above exists to produce
             the detection signal, not to BE the only training the segment
             gets.
          3. Fold (self._plateau_fold(), models/sketchlora.py -- merge
             mechanics (_compress()) completely unchanged, per explicit
             instruction), then detector.reset() (NOT part of the merge
             code -- the detector's own lifecycle, called from here), then
             open the next segment (_stream_begin_chunk).
        Real-task eval checkpoints are decoupled from fold timing exactly as
        in stream_run() (fire the moment cumulative images cross a real
        task's cumulative count) -- just checked after every batch instead
        of after every fixed chunk, since segments have no fixed size here.

        2026-11-02 REDESIGN: detector.reset() now runs after EVERY fire (a
        reversal from the original port, which deliberately never reset --
        see utils/online_lora_plateau.py's module docstring for the full
        story: a 6-task diagnostic run on this exact backbone showed the
        original design could get permanently stuck disarmed, collapsing
        CIL from 72.87% to 0.73% over 5 tasks with zero folds firing. Each
        segment now measures its own fresh baseline (EMA-smoothed window,
        `warmup` phase) instead of comparing against a baseline inherited
        from the last fire.
        """
        assert args["model_name"] in ("sketchlora", "sketchlora_align"), \
            "boundary_mode='loss_plateau' is a SketchLoRA-specific extension -- every other " \
            "method's _stream_begin_chunk/_stream_end_chunk overrides assume a fully-" \
            "materialized fixed-size chunk loader (e.g. InfLoRA's DualGPM passes), which this " \
            "driver's online phase never builds. Got model_name={}".format(args["model_name"])
        assert getattr(self, "lazy_merge_mode", "off") == "off", \
            "boundary_mode='loss_plateau' decides fold timing itself via LossPlateauDetector -- " \
            "lazy_merge_mode must stay 'off' (the default) so the two fold-deciders can't race " \
            "over the same _compress() call. Got lazy_merge_mode={}".format(self.lazy_merge_mode)
        assert getattr(self, "svd_period", 1) == 1, \
            "plateau_run() folds unconditionally on every detector fire (no period-skipping) -- " \
            "does not combine with svd_period>1. Got svd_period={}".format(self.svd_period)

        from utils.online_lora_plateau import LossPlateauDetector

        self.data_manager = data_manager
        epochs = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[plateau] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        seed_arg = args.get("seed", 1993)
        seed0 = seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg

        self._task_ranges, all_data, all_targets, task_img_cumends = \
            _build_stream_order(data_manager, seed0, _n_run)
        total_images = len(all_targets)

        detector = LossPlateauDetector(
            window_length=args.get("plateau_loss_window_length", 5),
            ema_beta=args.get("plateau_ema_beta", 0.4),
            variance_ratio=args.get("plateau_variance_ratio", 1.0),
            max_chunk_samples=args.get("plateau_max_chunk_samples", 2000))
        logging.info("[plateau] window_length={} ema_beta={} variance_ratio={} "
                     "max_chunk_samples={}; {} total images, {} tasks".format(
                         detector.window_length, detector.ema_beta,
                         detector.variance_ratio, detector.max_chunk_samples,
                         total_images, _n_run))

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        train_set = data_manager.get_dataset(
            [], source="train", mode="train", appendent=(all_data, all_targets))
        stream_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                   num_workers=num_workers)
        self._network.to(self._device)

        last_task_ended = -1
        results = []
        n_fires = 0
        seg_start = 0
        pos = 0
        seg_just_opened = True   # becomes True again after every fold
        loss_trace = []          # [(pos, ce_loss)] -- for OFFLINE threshold-tuning
                                  # (replay this fixed trace through LossPlateauDetector
                                  # with different thresholds without re-running training)
        fire_log = []             # [(pos, reason)]

        import time as _time
        _online_t0 = _time.time()
        _online_n_batches = 0

        for _, inputs, targets in stream_loader:
            _online_n_batches += 1
            bsz = inputs.size(0)
            tnp = targets.numpy()
            lo, hi = int(tnp.min()), int(tnp.max()) + 1
            if hi > self._total_classes:
                self._total_classes = hi
                self._network.update_fc(self._total_classes)
            # head growth above must run before _stream_begin_chunk, same invariant
            # stream_run() keeps for every chunk (fc must exist before freeze_to_task
            # touches fc.parameters()) -- the very first segment has no batch at all
            # to grow the head from until this point, unlike stream_run()'s fully
            # precomputed chunks.
            if seg_just_opened:
                self._stream_begin_chunk(None)
                seg_just_opened = False

            self._network.train()
            slot, merge = self._stream_slot(), self._stream_train_merge()
            inputs_d, targets_d = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs_d, task=slot, merge=merge)["logits"]
            ce_loss = F.cross_entropy(logits[:, lo:hi], targets_d - lo)
            extra = self._stream_extra_loss(lo, hi)
            loss = ce_loss if (isinstance(extra, float) and extra == 0.0) else ce_loss + extra
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()

            pos += bsz
            loss_value = ce_loss.detach().item()
            loss_trace.append((pos, bsz, loss_value))
            fired = detector.step(loss_value, bsz)

            if fired:
                n_fires += 1
                fire_log.append((pos, detector.last_fire_reason))
                seg_data = all_data[seg_start:pos]
                seg_targets = all_targets[seg_start:pos]
                if epochs > 1:
                    seg_lo, seg_hi = int(seg_targets.min()), int(seg_targets.max()) + 1
                    seg_set = data_manager.get_dataset(
                        [], source="train", mode="train", appendent=(seg_data, seg_targets))
                    seg_loader = DataLoader(seg_set, batch_size=self.batch_size, shuffle=True,
                                            num_workers=num_workers, persistent_workers=True)
                    for _ep in range(epochs - 1):
                        self._stream_train_epoch(seg_loader, seg_lo, seg_hi)
                # log base_mean/base_std BEFORE reset() clears them -- needed to tell
                # whether a fire's own bar was loose (e.g. segment 0, measured while
                # loss was still descending) or tight (e.g. a near-lossless merge's
                # warmup landing on an already-quiet stretch), per 2026-11-02 guidance
                # to observe this via logging rather than engineer around it blindly.
                logging.info("[plateau] fire #{} reason={} at image {} (segment {} images, "
                             "window mean={:.3f} var={:.3f}, base_mean={} base_std={})".format(
                                 n_fires, detector.last_fire_reason, pos, pos - seg_start,
                                 detector.last_window_mean, detector.last_window_variance,
                                 detector.base_mean, detector.base_std))
                self._plateau_fold()
                detector.reset()
                seg_start = pos
                seg_just_opened = True

            newly_completed = []
            t = last_task_ended + 1
            while t < _n_run and task_img_cumends[t] <= pos:
                newly_completed.append(t)
                t += 1
            for t in newly_completed:
                self._stream_task_to_chunk[t] = self._stream_slot()
                self._stream_end_task(t)
            if newly_completed:
                last_task_ended = newly_completed[-1]
                results.append(self._stream_eval(last_task_ended + 1))
                _stream_checkpoint_write(args, results, tag="plateau", partial=True)

        _online_elapsed = _time.time() - _online_t0
        logging.info("[DIAG plateau online-pass TOTAL] batches={} images={} elapsed={:.3f}s "
                     "({:.4f}s/image, {:.2f} images/s) -- includes retroactive-epoch time "
                     "(interleaved in the same wall clock, not separable here)".format(
                         _online_n_batches, pos, _online_elapsed,
                         _online_elapsed / max(pos, 1), pos / max(_online_elapsed, 1e-9)))
        logging.info("[plateau] total fires: {} over {} images ({} tasks)".format(
            n_fires, total_images, _n_run))

        seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed")
        trace_path = "run_logs/plateau_trace_{}_{}_s{}.json".format(
            args["model_name"], args.get("prefix", "run"), seed)
        with open(trace_path, "w") as f:
            json.dump({
                "task_img_cumends": task_img_cumends,
                "loss_trace": loss_trace,          # [(pos, batch_size, ce_loss)]
                "fires": fire_log,                 # [(pos, reason)]
                "detector_args": {
                    "window_length": detector.window_length,
                    "ema_beta": detector.ema_beta,
                    "variance_ratio": detector.variance_ratio,
                    "max_chunk_samples": detector.max_chunk_samples,
                },
            }, f)
        logging.info("[plateau] wrote loss trace + fire log to {} (for offline DIAGNOSIS "
                     "only -- replay loss_trace through LossPlateauDetector to inspect "
                     "behavior; per 2026-11-02 guidance this is not to be used to pick "
                     "final threshold values by maximizing alignment to known task "
                     "boundaries -- those should be set once, in advance)".format(trace_path))

        self._stream_results = results
        return results

    def buffered_plateau_run(self, data_manager, args):
        """SketchLoRA-only boundary mode, 2026-11-03 -- REPLACES plateau_run()'s
        deferred-retroactive-epoch design with a continuous, bounded-replay
        design modeled directly on Online_LoRA_Ref/Disjoint's own training
        loop (confirmed via direct code audit this session). NOT a variant
        of plateau_run() and does not call it or share its loop -- a fully
        separate codepath, per explicit instruction, so this stays isolated
        from plateau_run()/stream_run()/spectral_plateau_run() and cannot
        regress anything currently relied on.

        WHY plateau_run() was replaced rather than patched (see conversation
        record for the full derivation): plateau_run() withheld a segment's
        "real" (shuffled, tuned_epoch-1 extra) training until its detector
        fired, creating a deferred, bursty training structure with no
        equivalent in Online-LoRA's own reference -- this was directly
        responsible for both (a) an eval-timing mismatch (a real-task-
        boundary eval could land on a residual that hadn't received its
        deferred training yet) and (b) effectively naive single-pass SGD for
        long stretches, since nothing Online-LoRA actually relies on for
        stability (continuous replay) was ever ported alongside the detector.

        Per-step algorithm (every batch, unconditionally, no deferred phase):
          1. Forward the new batch; LOCAL-window CE (logits[:, lo:hi], targets
             - lo, reduction='mean') -- same windowing convention every other
             boundary mode in this codebase uses for its own "new data" loss.
          2. If the hard buffer is non-empty: forward its current contents
             too, but with a GLOBAL CIL-style softmax (logits[:, :total_
             classes], raw un-windowed targets) instead of a local window --
             deliberate, since buffer items can come from ANY past task's
             class range, not necessarily inside the current batch's [lo,hi)
             window (a single local window cannot cover both groups at once),
             and this matches what the buffer item is actually FOR: "can the
             model still tell this apart from everything it's seen," not "can
             it tell it apart from one arbitrary batch's local window."
          3. total_loss = local_new_loss + (global_buffer_loss if buffer non-
             empty else 0) + self._stream_extra_loss(lo, hi); ONE zero_grad/
             backward/step on total_loss, immediately, every batch. No
             epochs, no retroactive pass -- this IS the only training event.
          4. Feed total_loss (not just the new-batch component) into the
             SAME LossPlateauDetector used by plateau_run() (unchanged
             module, different instance) -- matches the reference, which
             also snapshots its combined (new + buffer) loss for its own
             loss-window detector, not the new batch's loss in isolation.
          5. Rebuild the hard buffer (utils/online_hard_buffer.py) from
             {buffer's own items, freshly re-scored under the CURRENT model
             this same step ∪ this step's own batch, per-sample losses},
             keeping the top `plateau_buffer_size` (default 4, matching the
             reference's own ImageNet-R default) by loss. Never cleared or
             reset at a fold -- see that module's docstring for why.
          6. If the detector fires: fold IMMEDIATELY (self._plateau_fold()),
             then detector.reset(), then open the next segment
             (_stream_begin_chunk). No deferred/retroactive training of any
             kind -- the fold is pure bookkeeping, exactly like Online-LoRA's
             own update_and_reset_lora_parameters(), which is why real-task-
             boundary eval (step 7) is now always safe to check after every
             batch: there is no "hasn't been trained yet" state left for it
             to land on.
          7. Real-task eval checkpoints: identical mechanics to plateau_run()
             (fire the moment cumulative images cross a real task's
             cumulative count, checked after every batch) -- this is the
             SAME code shape that was unsafe under plateau_run()'s deferred
             design, but is safe here since nothing is ever deferred.

        NOT ported (explicit, same original 2026-10-30 scope decision):
        Online-LoRA's MAS/EWC-style importance-weight regularizer. Only the
        detection state machine and the hard-buffer replay are in scope.

        `self.epochs`/`tuned_epoch` is UNUSED by this method -- there is no
        multi-epoch phase of any kind here, by design; every image gets
        exactly the continuous per-step training described above, nothing
        more, matching the reference's own effectively-single-exposure-plus-
        buffer-replay structure.
        """
        assert args["model_name"] in ("sketchlora", "sketchlora_align"), \
            "boundary_mode='loss_plateau_buffer' is a SketchLoRA-specific extension -- see " \
            "plateau_run()'s identical assert for the full reasoning. Got model_name={}".format(
                args["model_name"])
        assert getattr(self, "lazy_merge_mode", "off") == "off", \
            "boundary_mode='loss_plateau_buffer' decides fold timing itself via " \
            "LossPlateauDetector -- lazy_merge_mode must stay 'off'. Got lazy_merge_mode={}".format(
                self.lazy_merge_mode)
        assert getattr(self, "svd_period", 1) == 1, \
            "buffered_plateau_run() folds unconditionally on every detector fire (no period-" \
            "skipping) -- does not combine with svd_period>1. Got svd_period={}".format(
                self.svd_period)

        from utils.online_lora_plateau import LossPlateauDetector
        from utils.online_hard_buffer import HardExemplarBuffer

        self.data_manager = data_manager
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[plateau_buffer] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        seed_arg = args.get("seed", 1993)
        seed0 = seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg

        self._task_ranges, all_data, all_targets, task_img_cumends = \
            _build_stream_order(data_manager, seed0, _n_run)
        total_images = len(all_targets)

        detector = LossPlateauDetector(
            window_length=args.get("plateau_loss_window_length", 5),
            ema_beta=args.get("plateau_ema_beta", 0.4),
            variance_ratio=args.get("plateau_variance_ratio", 1.0),
            max_chunk_samples=args.get("plateau_max_chunk_samples", 2000))
        buffer_size = args.get("plateau_buffer_size", 4)
        buffer = HardExemplarBuffer(buffer_size=buffer_size)
        logging.info("[plateau_buffer] window_length={} ema_beta={} variance_ratio={} "
                     "max_chunk_samples={} buffer_size={}; {} total images, {} tasks".format(
                         detector.window_length, detector.ema_beta, detector.variance_ratio,
                         detector.max_chunk_samples, buffer_size, total_images, _n_run))

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        train_set = data_manager.get_dataset(
            [], source="train", mode="train", appendent=(all_data, all_targets))
        stream_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                   num_workers=num_workers)
        self._network.to(self._device)

        last_task_ended = -1
        results = []
        n_fires = 0
        pos = 0
        seg_just_opened = True
        loss_trace = []    # [(pos, bsz, combined_loss)] -- offline diagnosis, same shape as plateau_run()
        fire_log = []      # [(pos, reason)]

        for _, inputs, targets in stream_loader:
            bsz = inputs.size(0)
            tnp = targets.numpy()
            lo, hi = int(tnp.min()), int(tnp.max()) + 1
            if hi > self._total_classes:
                self._total_classes = hi
                self._network.update_fc(self._total_classes)
            if seg_just_opened:
                self._stream_begin_chunk(None)
                seg_just_opened = False

            self._network.train()
            slot, merge = self._stream_slot(), self._stream_train_merge()
            inputs_d, targets_d = inputs.to(self._device), targets.to(self._device)

            logits_new = self._network(inputs_d, task=slot, merge=merge)["logits"]
            per_sample_new = F.cross_entropy(logits_new[:, lo:hi], targets_d - lo, reduction="none")

            buf_inputs, buf_targets = buffer.replay_batch()
            per_sample_buf = None
            if buf_inputs is not None:
                buf_inputs, buf_targets = buf_inputs.to(self._device), buf_targets.to(self._device)
                logits_buf = self._network(buf_inputs, task=slot, merge=merge)["logits"]
                per_sample_buf = F.cross_entropy(logits_buf[:, :self._total_classes], buf_targets,
                                                 reduction="none")

            # Per-sample EQUAL weighting, matching Online-LoRA's Disjoint reference exactly
            # (engine.py:117-132: a flat, unnormalized sum over every individual new-batch AND
            # buffer loss, no per-group mean) -- NOT two separate group means summed at equal
            # weight (that gave each of the ~4 buffer samples ~12x the gradient weight of each
            # of the 48 new-batch samples, confirmed via a 3-task smoke test to cause loss
            # divergence: CE stuck at 9-26 instead of decreasing). One mean over the concatenated
            # per-sample losses makes every individual sample count equally regardless of pool,
            # mean-normalized (not Disjoint's literal raw sum) to stay batch-size-invariant like
            # every other loss in this codebase -- see utils/online_hard_buffer.py's own
            # docstring for the same reasoning applied to the buffer module itself.
            all_per_sample = per_sample_new if per_sample_buf is None else \
                torch.cat([per_sample_new, per_sample_buf])
            total_loss = all_per_sample.mean()

            extra = self._stream_extra_loss(lo, hi)
            if not (isinstance(extra, float) and extra == 0.0):
                total_loss = total_loss + extra

            self._stream_optim.zero_grad()
            total_loss.backward()
            self._stream_optim.step()

            buffer.update(inputs_d.detach(), targets_d.detach(), per_sample_new.detach(),
                         buffer_losses=(per_sample_buf.detach() if per_sample_buf is not None else None))

            pos += bsz
            loss_value = total_loss.detach().item()
            loss_trace.append((pos, bsz, loss_value))
            fired = detector.step(loss_value, bsz)

            if fired:
                n_fires += 1
                fire_log.append((pos, detector.last_fire_reason))
                logging.info("[plateau_buffer] fire #{} reason={} at image {} (buffer_len={}, "
                             "window mean={:.3f} var={:.3f}, base_mean={} base_std={})".format(
                                 n_fires, detector.last_fire_reason, pos, len(buffer),
                                 detector.last_window_mean, detector.last_window_variance,
                                 detector.base_mean, detector.base_std))
                self._plateau_fold()
                detector.reset()
                seg_just_opened = True

            newly_completed = []
            t = last_task_ended + 1
            while t < _n_run and task_img_cumends[t] <= pos:
                newly_completed.append(t)
                t += 1
            for t in newly_completed:
                self._stream_task_to_chunk[t] = self._stream_slot()
                self._stream_end_task(t)
            if newly_completed:
                last_task_ended = newly_completed[-1]
                results.append(self._stream_eval(last_task_ended + 1))
                _stream_checkpoint_write(args, results, tag="plateau_buffer", partial=True)

        logging.info("[plateau_buffer] total fires: {} over {} images ({} tasks)".format(
            n_fires, total_images, _n_run))

        seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed")
        trace_path = "run_logs/plateau_buffer_trace_{}_{}_s{}.json".format(
            args["model_name"], args.get("prefix", "run"), seed)
        with open(trace_path, "w") as f:
            json.dump({
                "task_img_cumends": task_img_cumends,
                "loss_trace": loss_trace,
                "fires": fire_log,
                "detector_args": {
                    "window_length": detector.window_length,
                    "ema_beta": detector.ema_beta,
                    "variance_ratio": detector.variance_ratio,
                    "max_chunk_samples": detector.max_chunk_samples,
                },
                "buffer_size": buffer_size,
            }, f)
        logging.info("[plateau_buffer] wrote loss trace + fire log to {}".format(trace_path))

        self._stream_results = results
        return results

    def spectral_plateau_run(self, data_manager, args):
        """SketchLoRA-only boundary mode, sibling to plateau_run() above --
        identical segment lifecycle and identical hysteresis state machine
        (utils.online_lora_plateau.LossPlateauDetector, reused UNCHANGED: its
        logic is signal-agnostic despite the name -- EMA-smooth a scalar,
        track a per-segment warmup baseline, fire when the windowed mean/
        variance drop back near that baseline), but the scalar fed into the
        detector is NOT the training CE loss -- it's a periodic read of one
        specific adapter's own "expected spectral energy loss"
        (sigma_{r_hat+1}^2 of layer `spectral_plateau_layer`'s
        `spectral_plateau_proj` projection, models/sketchlora.py::
        _probe_sigma_next_sq(), measured live without compressing/committing
        anything) instead of the loss-window LossPlateauDetector was
        originally built for.

        Design context (2026-11-02, following the CIFAR-100 streamsample
        diagnostics showing a clean boundary-synced sawtooth in this exact
        module -- see sketchlora_streamsample_wave1_cifar100_layer12v_*
        figures): layer 12's V-projection adapter was the cleanest single
        (layer, proj) signal of all 24 modules on CIFAR-100, which is why it
        is the default here (spectral_plateau_layer=12, spectral_plateau_
        proj="v") -- NOT assumed to transfer to ImageNet-R without checking;
        that is exactly what the smoke/trace configs below are for.

        Probe cadence: every `spectral_plateau_probe_every_n_batches` batches
        (default 1, i.e. every batch -- cheap enough at single-module scope
        that there is no real need to skip batches, see _probe_sigma_next_sq's
        own docstring; exposed as a knob anyway since the request that led to
        this was explicitly framed as "every batch, or every n batches").
        detector.step() is only called on probe batches -- the
        max_chunk_samples safety cap can therefore overshoot by up to
        (n-1)*batch_size images between probes; acceptable at n's expected
        small values (1-5) against a cap in the thousands.

        Still restricted to svd_period==1 / lazy_merge_mode=="off" (same
        reasons as plateau_run()) plus merge_op != "freqdir" (freqdir's whole
        point is to never form the dense delta_W this probe reads -- would
        silently defeat that ablation rather than error, so caught here
        explicitly).
        """
        assert args["model_name"] in ("sketchlora", "sketchlora_align"), \
            "boundary_mode='spectral_plateau' is a SketchLoRA-specific extension. " \
            "Got model_name={}".format(args["model_name"])
        assert getattr(self, "lazy_merge_mode", "off") == "off", \
            "boundary_mode='spectral_plateau' decides fold timing itself -- lazy_merge_mode " \
            "must stay 'off'. Got lazy_merge_mode={}".format(self.lazy_merge_mode)
        assert getattr(self, "svd_period", 1) == 1, \
            "spectral_plateau_run() folds unconditionally on every detector fire (no period-" \
            "skipping) -- does not combine with svd_period>1. Got svd_period={}".format(self.svd_period)
        assert self.merge_op != "freqdir", \
            "merge_op='freqdir' never forms the dense delta_W _probe_sigma_next_sq() reads -- " \
            "would silently defeat that ablation's entire point."

        from utils.online_lora_plateau import LossPlateauDetector

        self.data_manager = data_manager
        epochs = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[spectral_plateau] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        seed_arg = args.get("seed", 1993)
        seed0 = seed_arg[0] if isinstance(seed_arg, (list, tuple)) else seed_arg

        self._task_ranges, all_data, all_targets, task_img_cumends = \
            _build_stream_order(data_manager, seed0, _n_run)
        total_images = len(all_targets)

        probe_layer = args.get("spectral_plateau_layer", 12)       # 1-indexed ViT block
        probe_proj = args.get("spectral_plateau_proj", "v")
        probe_every_n = args.get("spectral_plateau_probe_every_n_batches", 1)
        probe_block_idx = probe_layer - 1

        detector = LossPlateauDetector(
            window_length=args.get("spectral_plateau_window_length", 5),
            ema_beta=args.get("spectral_plateau_ema_beta", 0.4),
            variance_ratio=args.get("spectral_plateau_variance_ratio", 1.0),
            max_chunk_samples=args.get("spectral_plateau_max_chunk_samples", 2000))
        logging.info("[spectral_plateau] layer={} proj={} probe_every_n_batches={} "
                     "window_length={} ema_beta={} variance_ratio={} max_chunk_samples={}; "
                     "{} total images, {} tasks".format(
                         probe_layer, probe_proj, probe_every_n,
                         detector.window_length, detector.ema_beta,
                         detector.variance_ratio, detector.max_chunk_samples,
                         total_images, _n_run))

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        train_set = data_manager.get_dataset(
            [], source="train", mode="train", appendent=(all_data, all_targets))
        stream_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False,
                                   num_workers=num_workers)
        self._network.to(self._device)

        last_task_ended = -1
        results = []
        n_fires = 0
        seg_start = 0
        pos = 0
        batches_since_probe = 0
        images_since_probe = 0
        seg_just_opened = True
        probe_trace = []          # [(pos, probe_value)] -- for OFFLINE threshold-tuning
        fire_log = []             # [(pos, reason)]

        for batch_idx, (_, inputs, targets) in enumerate(stream_loader):
            bsz = inputs.size(0)
            tnp = targets.numpy()
            lo, hi = int(tnp.min()), int(tnp.max()) + 1
            if hi > self._total_classes:
                self._total_classes = hi
                self._network.update_fc(self._total_classes)
            if seg_just_opened:
                self._stream_begin_chunk(None)
                seg_just_opened = False

            self._network.train()
            slot, merge = self._stream_slot(), self._stream_train_merge()
            inputs_d, targets_d = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs_d, task=slot, merge=merge)["logits"]
            ce_loss = F.cross_entropy(logits[:, lo:hi], targets_d - lo)
            extra = self._stream_extra_loss(lo, hi)
            loss = ce_loss if (isinstance(extra, float) and extra == 0.0) else ce_loss + extra
            self._stream_optim.zero_grad()
            loss.backward()
            self._stream_optim.step()

            pos += bsz
            batches_since_probe += 1
            images_since_probe += bsz
            fired = False
            if batches_since_probe >= probe_every_n:
                probe_value = self._probe_sigma_next_sq(probe_block_idx, probe_proj)
                probe_trace.append((pos, probe_value))
                fired = detector.step(probe_value, images_since_probe)
                batches_since_probe = 0
                images_since_probe = 0

            if fired:
                n_fires += 1
                fire_log.append((pos, detector.last_fire_reason))
                seg_data = all_data[seg_start:pos]
                seg_targets = all_targets[seg_start:pos]
                if epochs > 1:
                    seg_lo, seg_hi = int(seg_targets.min()), int(seg_targets.max()) + 1
                    seg_set = data_manager.get_dataset(
                        [], source="train", mode="train", appendent=(seg_data, seg_targets))
                    seg_loader = DataLoader(seg_set, batch_size=self.batch_size, shuffle=True,
                                            num_workers=num_workers, persistent_workers=True)
                    for _ep in range(epochs - 1):
                        self._stream_train_epoch(seg_loader, seg_lo, seg_hi)
                logging.info("[spectral_plateau] fire #{} reason={} at image {} (segment {} images, "
                             "window mean={:.4g} var={:.4g}, base_mean={} base_std={})".format(
                                 n_fires, detector.last_fire_reason, pos, pos - seg_start,
                                 detector.last_window_mean, detector.last_window_variance,
                                 detector.base_mean, detector.base_std))
                self._plateau_fold()
                detector.reset()
                seg_start = pos
                seg_just_opened = True

            newly_completed = []
            t = last_task_ended + 1
            while t < _n_run and task_img_cumends[t] <= pos:
                newly_completed.append(t)
                t += 1
            for t in newly_completed:
                self._stream_task_to_chunk[t] = self._stream_slot()
                self._stream_end_task(t)
            if newly_completed:
                last_task_ended = newly_completed[-1]
                results.append(self._stream_eval(last_task_ended + 1))
                _stream_checkpoint_write(args, results, tag="spectral_plateau", partial=True)

        logging.info("[spectral_plateau] total fires: {} over {} images ({} tasks)".format(
            n_fires, total_images, _n_run))

        seed = args["seed"][0] if isinstance(args.get("seed"), (list, tuple)) else args.get("seed")
        trace_path = "run_logs/spectral_plateau_trace_{}_{}_s{}.json".format(
            args["model_name"], args.get("prefix", "run"), seed)
        with open(trace_path, "w") as f:
            json.dump({
                "task_img_cumends": task_img_cumends,
                "probe_trace": probe_trace,        # [(pos, sigma_next_sq)]
                "fires": fire_log,                 # [(pos, reason)]
                "detector_args": {
                    "layer": probe_layer, "proj": probe_proj,
                    "probe_every_n_batches": probe_every_n,
                    "window_length": detector.window_length,
                    "ema_beta": detector.ema_beta,
                    "variance_ratio": detector.variance_ratio,
                    "max_chunk_samples": detector.max_chunk_samples,
                },
            }, f)
        logging.info("[spectral_plateau] wrote probe trace + fire log to {}".format(trace_path))

        self._stream_results = results
        return results

    # ---- legacy epoch-clock reconstruction (2026-07-20) ----
    def legacy_epoch_clock_run(self, data_manager, args):
        """Reconstruction of the ORIGINAL epoch-count-clock streaming design
        (boundary_mode="sample" + "boundary_mult", used for the 2026-07-03
        SVDLoRA/O-LoRA/InfLoRA/SeqLoRA comparison on cifar224 20t -- see
        run_logs/svdlora_cifar20t_sample_s1993.out). That implementation was
        superseded twice since (first by a sample-count clock, then by
        stream_run()'s unique-image-budget design above) and no longer exists
        in the codebase in its original form -- NOT a verified byte-exact
        restoration. Reconstructed from the one surviving log line
        ("[stream] adapter event every N global epochs (C=<mult> x <epochs>
        epochs)") plus BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's design recap.
        Specific mechanics that could NOT be confirmed and are therefore best-
        effort choices, not restorations: whether the threshold check could
        fire mid-epoch/mid-batch (implemented here as end-of-epoch only, the
        one granularity rule the log explicitly documents, albeit for the
        LATER sample-count redesign, not confirmed for this exact version);
        tail-of-run handling once the final task's epochs are mid-flight.

        Structurally the OPPOSITE of stream_run() above: TASK-MAJOR, not
        chunk-major. Real tasks train strictly in order, each with its own
        ordinary, single-task-only epoch loop and standard CIL class range --
        no image from two different tasks is EVER in the same batch, unlike
        stream_run()'s chunk-major design. The ONLY thing decoupled from real
        task boundaries is WHEN each method's own adapter bookkeeping
        (_stream_begin_chunk/_stream_end_chunk, unchanged, reused as-is) fires:
        instead of firing at every task boundary, it fires when a cumulative
        counter crosses multiples of a fixed threshold (derived from
        stream_budget_mb purely to give this historical clock a size in the
        same MB units as the current design, for direct comparison) -- and
        that counter uses EPOCH-REPEATED counting (an epoch trained over a
        task's N images adds N every time, so re-running epochs over the same
        task DOES inflate the counter) -- this is the exact "epoch-repeat"
        behavior stream_run()'s 2026-07-19 redesign was built to replace, so
        reconstructing it here is deliberate, not a regression.
        """
        epochs = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[legacy] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))
        stream_budget_mb = float(args["stream_budget_mb"])
        budget_images = max(1, round(stream_budget_mb * 1024 * 1024 / BYTES_PER_IMAGE))

        self.data_manager = data_manager
        self._task_ranges = []
        known = 0
        for t in range(_n_run):
            task_size = data_manager.get_task_size(t)
            self._task_ranges.append((known, known + task_size))
            known += task_size

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        logging.info("[legacy] adapter event every {} epoch-repeated samples ({}MB-equivalent)".format(
            budget_images, stream_budget_mb))

        cumulative_samples = 0
        folds_fired = 0
        results = []
        slot_opened = False

        for t in range(_n_run):
            lo, hi = self._task_ranges[t]
            if hi > self._total_classes:
                self._total_classes = hi
                self._network.update_fc(self._total_classes)
            task_data, task_targets, _ = data_manager.get_dataset(
                np.arange(lo, hi), source="train", mode="train", ret_data=True)
            train_set = data_manager.get_dataset(
                [], source="train", mode="train", appendent=(task_data, task_targets))
            # persistent_workers=True: reused across `epochs` iterations below, same
            # reasoning as stream_run()'s own per-chunk loader above.
            loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, persistent_workers=True)
            self._network.to(self._device)

            if not slot_opened:
                self._stream_begin_chunk(loader)
                slot_opened = True

            for ep in range(epochs):
                self._stream_train_epoch(loader, lo, hi)
                cumulative_samples += len(task_data)   # epoch-repeated counting (reconstructed)
                new_folds = cumulative_samples // budget_images
                while new_folds > folds_fired:
                    folds_fired += 1
                    self._stream_end_chunk(loader)
                    # "completed" = real tasks whose OWN epochs have fully finished by now
                    completed = t if ep < epochs - 1 else t + 1
                    self._stream_begin_chunk(loader)
                    if completed > 0:
                        self._stream_task_to_chunk[completed - 1] = self._stream_slot()
                        results.append(self._stream_eval(completed))

            self._stream_end_task(t)

        if not results or results[-1]["completed"] != _n_run:
            results.append(self._stream_eval(_n_run))

        self._stream_results = results
        return results

    # ---- offset-consolidation: fold at a configurable epoch cadence/phase,
    # decoupled from task boundaries ("Offset-consolidation protocol (E2).md",
    # 2026-11-04 -- mechanism only; the source doc's experimental "arms"/
    # invariant-auditing/multi-seed design is NOT implemented here, just the
    # driver that would let someone run any one (k, phi) setting). ----------
    def offset_consolidation_run(self, data_manager, args):
        """Folds the current transient into the sketch every `offset_k` epochs,
        phase-shifted by `offset_phi`, instead of exactly at every task
        boundary. Generalizes the implicit "aligned" case every oracle-mode
        2-slot sketch method already runs (k == epochs, phi == 0 -- fold at
        every task's end) to arbitrary (k, phi): k < epochs folds multiple
        times within one task; k > epochs lets one transient train across a
        task boundary without resetting. Reuses the exact same 3-hook
        contract every other driver in this file uses (_stream_slot /
        _stream_begin_chunk / _stream_end_chunk / _stream_cil_forward) -- any
        method that already plugs into stream_run()/legacy_epoch_clock_run()
        needs only those same hooks to work here too (see e.g.
        models/lorasub_sketch.py's _stream_* overrides).

        Epoch-slot timeline: slot s = 0..NE-1, task n(s) = s // E, where
        E = self.epochs (this method's per-task epoch budget) and N is
        however many tasks this run covers. Merge after slot s whenever
        (s + 1 - phi) % k == 0, plus unconditionally after the very last
        slot (so the run always ends on a folded state). A "segment" is the
        run of slots between two consecutive merge points -- one transient
        (one _stream_begin_chunk/_stream_end_chunk pair, one optimizer/LR
        schedule) trains across its ENTIRE segment; the loop below never
        resets the optimizer at a task boundary on its own, only at a
        computed merge point -- same persistence legacy_epoch_clock_run()
        already relies on for ITS (sample-count, not epoch-count) clock.

        LR SCHEDULE (corrected 2026-11-06, see the conversation record):
        each new transient's cosine schedule uses _stream_new_optimizer()'s
        own flat T_max=self.epochs -- the SAME per-task-epoch-budget
        constant every other oracle-mode method already anneals over --
        NOT resized to the real segment length. An earlier version of this
        method DID resize T_max per segment (via a since-removed
        _segment_len() helper); that was reverted after review: the
        scheduler is part of the method, and a schedule that sizes itself
        to the REAL upcoming segment length needs to know ahead of time
        exactly how many epochs remain until the next fold -- privileged
        knowledge about this experiment's external (k, phi) clock that the
        method's own fixed hyperparameters (E alone) do not give it, even
        though (k, phi, E) happen to be known constants at config time. A
        short segment is simply cut off before its schedule finishes
        annealing; a long one sits at the floor LR for the remainder --
        this is a real, intended cost of fold timing under this protocol,
        not something to engineer around.

        Evaluation happens at every TASK boundary only (not at every merge
        point -- a merge and a task boundary coincide only in the aligned
        case), using whatever the live state currently is: "W + sketch +
        the current unmerged transient" if no merge lands exactly there,
        or the just-folded state if one does. Deliberately evaluates BEFORE
        checking for a coincident merge, not after -- the source document's
        own pseudocode got this order backwards (evaluating after an
        aligned-case merge+reset would measure a lossy-compressed version of
        the task's knowledge, not what was actually trained, and wouldn't
        reproduce plain per-task sketch runs even at k=epochs/phi=0; see the
        document review for the full argument). No separate immediate
        pre/post-merge eval pair is logged -- that was the source document's
        Per-merge-logging diagnostic, explicitly out of scope here.
        """
        k = int(args["offset_k"])
        phi = int(args.get("offset_phi", 0))
        assert k > 0, "offset_k must be positive"
        assert 0 <= phi < k, "offset_phi must be in [0, offset_k)"
        E = self.epochs
        nb_tasks = data_manager.nb_tasks
        _n_run = args.get("stop_after_tasks") or nb_tasks
        _n_run = min(_n_run, nb_tasks)
        if _n_run < nb_tasks:
            logging.info("[offset] stop_after_tasks={} (of {} total)".format(_n_run, nb_tasks))

        total_slots = _n_run * E
        merge_after = {s for s in range(total_slots) if (s + 1 - phi) % k == 0}
        merge_after.add(total_slots - 1)
        merge_sorted = sorted(merge_after)
        logging.info("[offset] k={} phi={} E={} total_slots={} merge points at slots {}".format(
            k, phi, E, total_slots, merge_sorted))

        self.data_manager = data_manager
        self._task_ranges = []
        known = 0
        for t in range(_n_run):
            task_size = data_manager.get_task_size(t)
            self._task_ranges.append((known, known + task_size))
            known += task_size

        self._known_classes = 0
        self._total_classes = 0
        self._cur_task = -1
        self._stream_chunk = -1
        self._stream_task_to_chunk = {}
        self._stream_init()

        results = []
        slot_opened = False
        cur_task_idx = -1
        loader, lo, hi = None, 0, 0

        for s in range(total_slots):
            t = s // E
            if t != cur_task_idx:
                cur_task_idx = t
                lo, hi = self._task_ranges[t]
                if hi > self._total_classes:
                    self._total_classes = hi
                    self._network.update_fc(self._total_classes)
                task_data, task_targets, _ = data_manager.get_dataset(
                    np.arange(lo, hi), source="train", mode="train", ret_data=True)
                train_set = data_manager.get_dataset(
                    [], source="train", mode="train", appendent=(task_data, task_targets))
                # persistent_workers=True: this loader is reused across however
                # many slots of this task fall in the current segment.
                loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                    num_workers=num_workers, persistent_workers=True)
                self._network.to(self._device)

            if not slot_opened:
                self._stream_begin_chunk(loader)
                # NOT segment-length-aware (reverted 2026-11-06, see the
                # conversation record): _stream_new_optimizer() already
                # builds this scheduler with a flat T_max=self.epochs, the
                # SAME per-task-epoch-budget constant every other oracle-
                # mode method already uses to anneal -- a scheduler that
                # instead sized itself to the REAL upcoming segment length
                # would need to know ahead of time exactly how many epochs
                # remain until the next fold, which is privileged knowledge
                # about this experiment's external (k, phi) clock, not
                # something the method's own fixed hyperparameters (E
                # alone) determine. A short segment is simply cut off
                # before its schedule finishes annealing; a long one sits
                # at the floor LR for the remainder -- a real cost of fold
                # timing, not something to engineer away.
                slot_opened = True

            self._stream_train_epoch(loader, lo, hi)

            task_boundary = (s + 1) % E == 0
            if task_boundary:
                self._stream_task_to_chunk[t] = self._stream_slot()
                results.append(self._stream_eval(t + 1))

            if s in merge_after:
                self._stream_end_chunk(loader)
                slot_opened = False
                if s < total_slots - 1:
                    self._stream_begin_chunk(loader)   # fresh flat-T_max=self.epochs schedule
                    slot_opened = True

            if task_boundary:
                self._stream_end_task(t)

        # 2026-11-06 addition: forgetting. offset_consolidation_run (and
        # every other boundary_mode streaming driver, via the shared
        # _stream_eval above) previously had NO forgetting computation at
        # all -- _run_stream's own JSON-writer (trainer.py) only ever
        # logged the aggregate CIL/TIL curves, since _stream_eval returned
        # a single pooled accuracy, not a per-task breakdown that a
        # forgetting metric needs. Mirrors trainer.py's own oracle-mode
        # print_forget block EXACTLY (same matrix shape, same formula:
        # forgetting = mean over PAST tasks of (that task's peak-ever
        # accuracy - its accuracy at the final checkpoint)) so this is
        # directly comparable to every oracle-mode run's own "Forgetting
        # (CNN): X" line, not a different metric under the same name.
        if args.get("print_forget") and results:
            n_checkpoints = len(results)
            final_n_tasks = results[-1]["completed"]
            np_acctable = np.zeros([final_n_tasks, n_checkpoints])
            for col, r in enumerate(results):
                row = r["cil_per_task"]
                np_acctable[:len(row), col] = np.array(row)
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, -1])[:final_n_tasks - 1])
            logging.info("[offset] Accuracy Matrix (CNN, tasks x checkpoints):\n{}".format(
                np.array2string(np_acctable, max_line_width=100000, precision=2, suppress_small=True)))
            logging.info("Forgetting (CNN): {}".format(forgetting))
            results[-1]["forgetting"] = float(forgetting)

        self._stream_results = results
        return results

    # ---- streaming evaluation: completed tasks only, folded adapter ----
    @torch.no_grad()
    def _stream_eval(self, completed):
        """CIL (argmax over completed classes) on the first `completed` class-tasks,
        using the method's deployed adapter. TIL is meaningless in this memory-
        increment setup (user-confirmed 2026-07-19: the whole point is that task
        boundaries/counts aren't available, so a task-oracle masked eval doesn't
        belong here) -- OFF by default, computed only if a config explicitly sets
        "stream_til": true (kept, not deleted, in case a future diagnostic wants
        it; every generated streaming config omits the key, so this is CIL-only
        by default going forward)."""
        self._network.eval()
        compute_til = self.args.get("stream_til", False)
        ranges = self._task_ranges[:completed]
        hi_total = ranges[-1][1]
        cil_correct = cil_n = til_correct = til_n = 0
        til_per_task = []
        # 2026-11-06 addition: per-task-group CIL accuracy (the GLOBAL
        # argmax's prediction, restricted to each task's OWN held-out
        # images -- NOT task-routed/TIL) -- unconditional, independent of
        # stream_til, since this is what a forgetting computation needs
        # (trainer.py's own oracle-mode cnn_matrix/print_forget convention:
        # forgetting = mean over past tasks of (that task's PEAK accuracy
        # ever recorded - its accuracy at the current/final checkpoint)).
        # Was previously entirely absent from this method -- only the
        # aggregate `cil` (over ALL completed classes combined) was ever
        # computed, which cannot be decomposed back into per-task numbers
        # after the fact.
        cil_per_task = []
        for t, (lo, hi) in enumerate(ranges):
            ds = self.data_manager.get_dataset(np.arange(lo, hi), source="test", mode="test")
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
            t_corr, t_n = 0, 0
            cil_t_corr, cil_t_n = 0, 0
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device); tnp = targets.numpy()
                cil_logits = self._stream_cil_forward(inputs)["logits"][:, :hi_total]
                cil_pred = cil_logits.argmax(1).cpu().numpy()
                cil_correct += int((cil_pred == tnp).sum()); cil_n += len(tnp)
                cil_t_corr += int((cil_pred == tnp).sum()); cil_t_n += len(tnp)
                if compute_til:
                    til_logits = self._forward_task(inputs, t)["logits"][:, lo:hi]
                    til_pred = til_logits.argmax(1).cpu().numpy() + lo
                    c = int((til_pred == tnp).sum())
                    til_correct += c; til_n += len(tnp); t_corr += c; t_n += len(tnp)
            cil_per_task.append(round(100.0 * cil_t_corr / max(cil_t_n, 1), 2))
            if compute_til:
                til_per_task.append(round(100.0 * t_corr / max(t_n, 1), 2))
        cil = round(100.0 * cil_correct / max(cil_n, 1), 2)
        til = round(100.0 * til_correct / max(til_n, 1), 2) if compute_til else None
        # diagnostic-only (2026-07-19): current chunk/slot index (what
        # _stream_cil_forward actually deploys) alongside the real-task
        # checkpoint count, to directly measure how far apart the two clocks
        # have drifted -- see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's
        # ProgPrompt investigation. Purely additive logging, no behavior change.
        if compute_til:
            logging.info("[stream eval] completed {:>2} | chunk {:>3} | CIL {:.2f} | TIL {:.2f} | TIL/task {}".format(
                completed, self._stream_slot(), cil, til, til_per_task))
        else:
            logging.info("[stream eval] completed {:>2} | chunk {:>3} | CIL {:.2f}".format(
                completed, self._stream_slot(), cil))
        return {"completed": completed, "cil": cil, "til": til, "til_per_task": til_per_task,
                "cil_per_task": cil_per_task}
