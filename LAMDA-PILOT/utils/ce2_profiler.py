"""CE2: coarse, process-level computational-efficiency profiling (2026-08-10).

Deliberately much simpler than utils/ce_profiler.py's narrow-region/step-vs-
boundary scheme: exactly THREE torch.profiler sessions per task -- the whole
training loop, the whole eval, and the method's own boundary bookkeeping (e.g.
SketchLoRA's SVD compress, InfLoRA's DualGPM passes) -- with no per-step
sampling, no ce_region() tagging, and no split_by_recurrence scaling. Every op
that executes inside a session's window is harvested (via key_averages(), the
same technique utils/ops_ledger.py::measure_step_macs already uses for its
single-batch probe), not just explicitly-tagged sub-regions.

This trades precision (no attribution below "the whole training loop") for
uniformity: it needs no method-specific `_train_adapter`/`train_merge` concepts,
so it applies identically to all 9 in-scope methods, including the 4
(CL-LoRA/RainbowPrompt/TUNA/EASE) the old ce_profiler.py-based OpsLedger skips
entirely (trainer.py's `hasattr(model, "_train_adapter")` gate).

NOT re-entrancy-safe against utils/ce_profiler.py's CEProfileSession or
utils/ops_ledger.py's raw torch.profiler.profile() use (PyTorch does not
support concurrent profiler instances). trainer.py's ce2_enabled path is
responsible for ensuring the legacy machinery never activates alongside this
one -- see trainer.py's own comment at the ce2_enabled block for how that's
guaranteed (model._ce_boundary_ctrl/_ce_step_acc are simply never set).
"""
import json
import logging
import os
import time
from contextlib import contextmanager

import torch

_ZERO_TOTALS = {"wall_seconds": 0.0, "macs": 0.0, "device_seconds": 0.0, "host_seconds": 0.0}


def _harvest_totals(prof):
    """Sum MACs/device-seconds/host-seconds over EVERY op recorded in a
    finished profile -- no region-tag filtering (contrast utils/ce_profiler.py's
    _harvest, which only captures explicitly ce_region()-tagged spans). Uses
    "self" time (self_cuda_time_total / self_device_time_total /
    self_cpu_time_total), which is already exclusive of child-call time, so
    summing over every key-averaged event does not double-count nested calls.
    flops is likewise only populated by the profiler on actual compute-kernel
    ops (matmul/conv/etc.), never on container/module-call wrapper events, so
    the same "sum everything" approach is safe for MACs too."""
    try:
        avgs = prof.key_averages()
    except Exception as e:                                    # pragma: no cover
        logging.warning("[CE2 profiler] could not read profiler events: %s", e)
        return dict(macs=0.0, device_seconds=0.0, host_seconds=0.0)
    macs = 0.0
    device_us = 0.0
    host_us = 0.0
    for evt in avgs:
        flops = getattr(evt, "flops", None)
        if flops:
            macs += float(flops) / 2.0   # FLOPs -> MACs, same convention as ops_ledger.py
        self_device = getattr(evt, "self_cuda_time_total", None)
        if self_device is None:
            self_device = getattr(evt, "self_device_time_total", 0.0)
        device_us += float(self_device or 0.0)
        host_us += float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
    return dict(macs=macs, device_seconds=device_us / 1e6, host_seconds=host_us / 1e6)


class CE2Session:
    """One coarse profiling window: wall-clock + torch.profiler(with_flops=True)
    together, harvested as a single (macs, device_seconds, host_seconds) total
    for the whole window. `ok=False` (all-zero totals, wall_seconds still valid)
    if the profiler failed to start/harvest -- never raises, mirroring
    CEProfileSession's "a broken measurement must not kill a multi-hour run"
    convention."""

    def __init__(self, device):
        self.device = device
        self.wall_seconds = 0.0
        self.macs = 0.0
        self.device_seconds = 0.0
        self.host_seconds = 0.0
        self.ok = False
        self._prof = None
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if getattr(self.device, "type", None) == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            self._prof = torch.profiler.profile(activities=activities, with_flops=True)
            self._prof.__enter__()
        except Exception as e:                                # pragma: no cover
            logging.warning("[CE2 profiler] failed to start: %s", e)
            self._prof = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prof is not None:
            try:
                if getattr(self.device, "type", None) == "cuda":
                    torch.cuda.synchronize()
                self._prof.__exit__(exc_type, exc, tb)
                if exc_type is None:
                    totals = _harvest_totals(self._prof)
                    self.macs = totals["macs"]
                    self.device_seconds = totals["device_seconds"]
                    self.host_seconds = totals["host_seconds"]
                    self.ok = True
            except Exception as e:                            # pragma: no cover
                logging.warning("[CE2 profiler] harvest failed: %s", e)
            finally:
                self._prof = None
        self.wall_seconds = time.time() - self._t0
        return False

    def totals(self):
        return dict(wall_seconds=self.wall_seconds, macs=self.macs,
                     device_seconds=self.device_seconds, host_seconds=self.host_seconds)


class CE2Accumulator:
    """Sums possibly-multiple CE2Session windows into one running total for a
    single task's boundary bucket (e.g. InfLoRA's two separate DualGPM passes,
    or SketchLoRA-CA's compress+align, both need to land in the same bucket).
    Call reset() once per task (trainer.py, before incremental_train), then
    each method's own boundary call site opens sessions via ce2_boundary()."""

    def __init__(self, device):
        self.device = device
        self._totals = dict(_ZERO_TOTALS)

    def reset(self):
        self._totals = dict(_ZERO_TOTALS)

    def session(self):
        return CE2Session(self.device)

    def add(self, sess):
        self._totals["wall_seconds"] += sess.wall_seconds
        if sess.ok:
            self._totals["macs"] += sess.macs
            self._totals["device_seconds"] += sess.device_seconds
            self._totals["host_seconds"] += sess.host_seconds

    def totals(self):
        return dict(self._totals)


@contextmanager
def ce2_boundary(model):
    """Method-side entry point: `with ce2_boundary(self): self._compress()`
    (or whatever the method's own boundary op is). Plain passthrough (no
    profiler, zero overhead) whenever model._ce2_boundary_acc is unset -- i.e.
    a complete no-op on every run that doesn't set ce2_enabled -- so it is
    always safe to leave these wraps in method code permanently."""
    acc = getattr(model, "_ce2_boundary_acc", None)
    if acc is None:
        yield
        return
    with acc.session() as sess:
        yield
    acc.add(sess)


def diff_totals(total, subtract):
    """total - subtract, clamped at 0 per field (used to derive "pure train" =
    whole incremental_train() window minus the boundary bucket already carved
    out of it)."""
    return {k: max(0.0, total.get(k, 0.0) - subtract.get(k, 0.0)) for k in _ZERO_TOTALS}


class CE2Logger:
    """Writes one JSON per run under out_dir, one record per task (persisted
    incrementally, same "safe to read after a kill" convention as
    utils/metrics_logger.py), plus a final summary. Fields are the raw
    per-bucket totals (wall_seconds/macs/device_seconds/host_seconds) for
    train/eval/boundary -- CE-vs-SeqLoRA ratios are computed by a separate
    post-hoc script (scripts/compute_ce2_scores.py), not here, so this stays a
    pure recorder."""

    def __init__(self, out_dir, tag, args=None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, "ce2_{}.json".format(tag))
        self.meta = dict(
            model_name=(args or {}).get("model_name"),
            dataset=(args or {}).get("dataset"),
            seed=(args or {}).get("seed"),
            tuned_epoch=(args or {}).get("tuned_epoch"),
            batch_size=(args or {}).get("batch_size"),
        )
        self.tasks = []
        self._write()

    def record_task(self, task_idx, train, eval, boundary, cil_top1=None):
        self.tasks.append(dict(
            task=task_idx, cil_top1=cil_top1,
            train=train, eval=eval, boundary=boundary,
        ))
        self._write()

    def finalize(self):
        self._write()

    def _write(self):
        try:
            with open(self.out_path, "w") as f:
                json.dump(dict(meta=self.meta, tasks=self.tasks), f, indent=2)
        except Exception as e:                                # pragma: no cover
            logging.warning("[CE2 logger] failed to write %s: %s", self.out_path, e)
