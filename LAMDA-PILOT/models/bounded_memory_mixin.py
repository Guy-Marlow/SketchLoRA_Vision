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

num_workers = 8
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

    def _bounded_train_epoch(self, loader, optimizer, scheduler, cycle_class_mask):
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
        bounded_memory_harness.md for the recorded choice."""
        self._network.train()
        slot, merge = self._stream_slot(), self._stream_train_merge()
        for _, inputs, targets in loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs, task=slot, merge=merge)["logits"]
            masked_logits = logits + cycle_class_mask
            loss = F.cross_entropy(masked_logits, targets)
            extra = self._stream_extra_loss(0, logits.shape[1])
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

        results = []
        cum_images = 0
        next_ckpt_idx = 0
        cycle_idx = -1
        _prev_param_hash = None   # Round 2 §2.2 eval-routing identity check
        mlog.begin_task()   # starts timing the FIRST checkpoint-interval
        while cum_images < total_images:
            cycle_idx += 1
            c_start = cum_images
            c_end = min(total_images, c_start + cycle_images)
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

            self._stream_begin_chunk(loader)
            self._bounded_new_optimizer()   # Round-2 §1.2: head weight_decay=0, uniform
            optimizer, scheduler = self._stream_optim, self._stream_sched
            for _ep in range(epochs):
                self._bounded_train_epoch(loader, optimizer, scheduler, cycle_class_mask)
            self._stream_end_chunk(loader)

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
                if _prev_param_hash is not None:
                    assert param_hash != _prev_param_hash, (
                        "Round 2 §2.2 eval-routing identity check FAILED: parameter-state "
                        "hash did not change between consecutive checkpoints (cycle {} -> "
                        "{}) -- eval may be reading a stale/frozen state instead of the "
                        "just-trained one.".format(cycle_idx, cycle_idx))
                _prev_param_hash = param_hash
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
