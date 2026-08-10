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
    ``stream_budget_mb`` worth of images (224x224x3 bytes/image). A chunk that
    only partially overlaps a task carries the task's remaining images forward
    into the very next chunk -- no image is ever shown twice, and none are
    skipped.
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

# 2026-08-10: 8->4, see models/lora.py's identical change for rationale.
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
                    ("model_name", "dataset", "init_cls", "increment", "stream_budget_mb",
                     "n_lora_blocks", "init_lr", "svd_energy_target", "lamda_1", "lamb", "lame")},
                    "results": results, "partial": partial}, f, indent=2)

# uint8 model-input accounting: 224x224x3, the same "memory budget" convention
# used by utils/budget_stream.py -- kept identical so MB values mean the same
# thing across both streaming designs.
BYTES_PER_IMAGE = 224 * 224 * 3


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
        stream_budget_mb = float(args["stream_budget_mb"])   # required, no silent default
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
        self._task_ranges = []
        data_parts, targets_parts = [], []
        task_img_cumends = []   # cumulative UNIQUE image count through the end of each real task
        running = 0
        known = 0
        for t in range(_n_run):
            task_size = data_manager.get_task_size(t)
            lo, hi = known, known + task_size
            self._task_ranges.append((lo, hi))
            task_data, task_targets, _ = data_manager.get_dataset(
                np.arange(lo, hi), source="train", mode="train", ret_data=True)
            # fixed per-task permutation (mixes this task's own classes together) --
            # this task's own "arrival order" in the stream, seeded for reproducibility.
            perm = np.random.RandomState(seed0 * 9973 + t).permutation(len(task_data))
            data_parts.append(task_data[perm])
            targets_parts.append(task_targets[perm])
            running += len(task_data)
            task_img_cumends.append(running)
            known = hi
        all_data = np.concatenate(data_parts)
        all_targets = np.concatenate(targets_parts)
        total_images = len(all_targets)

        chunk_bounds = list(range(0, total_images, budget_images)) + [total_images]
        chunks = [(chunk_bounds[i], chunk_bounds[i + 1]) for i in range(len(chunk_bounds) - 1)
                  if chunk_bounds[i + 1] > chunk_bounds[i]]
        logging.info("[stream] unique-image budget {} images ({}MB); {} total images, "
                     "{} tasks -> {} chunks".format(
                         budget_images, stream_budget_mb, total_images, _n_run, len(chunks)))

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
            loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers)
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
            loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers)
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
        for t, (lo, hi) in enumerate(ranges):
            ds = self.data_manager.get_dataset(np.arange(lo, hi), source="test", mode="test")
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
            t_corr, t_n = 0, 0
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device); tnp = targets.numpy()
                cil_logits = self._stream_cil_forward(inputs)["logits"][:, :hi_total]
                cil_pred = cil_logits.argmax(1).cpu().numpy()
                cil_correct += int((cil_pred == tnp).sum()); cil_n += len(tnp)
                if compute_til:
                    til_logits = self._forward_task(inputs, t)["logits"][:, lo:hi]
                    til_pred = til_logits.argmax(1).cpu().numpy() + lo
                    c = int((til_pred == tnp).sum())
                    til_correct += c; til_n += len(tnp); t_corr += c; t_n += len(tnp)
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
        return {"completed": completed, "cil": cil, "til": til, "til_per_task": til_per_task}
