"""CE2: coarse, process-level computational-efficiency profiling (2026-08-10).

Deliberately much simpler than utils/ce_profiler.py's narrow-region/step-vs-
boundary scheme: exactly THREE measurement buckets per task -- the whole
training loop, the whole eval, and the method's own boundary bookkeeping (e.g.
SketchLoRA's SVD compress, InfLoRA's DualGPM passes) -- with no per-step
sampling and no split_by_recurrence scaling. Every op that executes inside a
window is harvested, not just explicitly-tagged sub-regions -- except for the
one "boundary" tag described below, which exists purely to split an already-
open window in two, not to pick out arbitrary named regions the way
ce_profiler.py's ce_region() does.

This trades precision (no attribution below "the whole training loop" / "the
boundary op") for uniformity: it needs no method-specific `_train_adapter`/
`train_merge` concepts, so it applies identically to all 9 in-scope methods,
including the 4 (CL-LoRA/RainbowPrompt/TUNA/EASE) the old ce_profiler.py-based
OpsLedger skips entirely (trainer.py's `hasattr(model, "_train_adapter")` gate).

NOT re-entrancy-safe against utils/ce_profiler.py's CEProfileSession or
utils/ops_ledger.py's raw torch.profiler.profile() use (PyTorch does not
support concurrent profiler instances). trainer.py's ce2_enabled path is
responsible for ensuring the legacy machinery never activates alongside this
one -- see trainer.py's own comment at the ce2_enabled block for how that's
guaranteed (model._ce_boundary_ctrl/_ce_step_acc are simply never set).

IS re-entrancy-safe against ITSELF, via the module-level _ACTIVE_SESSION guard
below -- fixed 2026-08-11 after a real, confirmed corruption found in
ce2_omnibenchmark1k_10t/olora.out: trainer.py's per-task "total" CE2Session
wraps the WHOLE model.incremental_train() call, and several methods
(O-LoRA/InfLoRA/TreeLoRA/CL-LoRA/EASE/TUNA/SketchLoRA) also open their OWN
CE2Session, via ce2_boundary(), for their boundary bookkeeping -- nested
INSIDE that same window. torch.profiler's Kineto backend is a single
process-wide singleton: opening a second torch.profiler.profile() while one is
already active corrupts the outer one's harvest at its own, later __exit__
("Can't disable Kineto profiler when it's not running"), silently zeroing its
macs/device_seconds/host_seconds. Confirmed by reproducing the identical
RuntimeError with two bare nested torch.profiler.profile() context managers
outside any training code.

FIX, in two parts:
  1. Only ONE real torch.profiler.profile() is ever open at a time (tracked by
     _ACTIVE_SESSION, a reference to whichever CE2Session currently holds it --
     always the outer "total" session, since it opens before
     incremental_train() runs). This alone would be enough to stop the crash,
     but a nested ce2_boundary() call would then have nothing to measure with
     and its bucket would read zero -- a real loss of the "how much does this
     method's boundary op specifically cost" question CE2 exists to answer for
     methods like SketchLoRA (SVD compress) and InfLoRA (DualGPM passes).
  2. So instead of falling back to a silent no-op, a NESTED ce2_boundary() call
     tags its region with torch.profiler.record_function(BOUNDARY_TAG) INSIDE
     the currently-active session's own already-open profiler (a lightweight
     marker, not a second profiler instance -- costs nothing Kineto-wise). At
     the active session's own harvest time, _split_totals() walks the finished
     profile's event tree once and partitions self-time/self-macs into
     "inside a BOUNDARY_TAG scope" vs "outside" (self_* fields are already
     exclusive of child-call time, so summing them over every event, tagged or
     not, root or nested, is double-count-free regardless of tree shape --
     this is the same invariant the old ce_profiler.py's ce_region()
     attribution relied on, just collapsed to a single tag instead of many
     named regions). The active session exposes both halves
     (macs/device_seconds/host_seconds for the whole window, PLUS
     boundary_macs/boundary_device_seconds/boundary_host_seconds for just the
     tagged portion); trainer.py folds the latter into that task's boundary
     bucket once the total session closes (see its ce2_enabled block). Wall-
     clock time for the nested call is still measured directly (a plain
     time.time() delta around the yield), independent of the tag/split
     machinery, since that never needed the profiler in the first place.

Net effect: the outer "total" bucket is always correct (never interfered
with), AND a nested boundary op's macs/device/host are recovered via tagging
rather than lost -- both the crash and the earlier boundary-attribution
regression are fixed together.
"""
import json
import logging
import os
import time
from contextlib import contextmanager

import torch

_ZERO_TOTALS = {"wall_seconds": 0.0, "macs": 0.0, "device_seconds": 0.0, "host_seconds": 0.0}

# Module-level reentrancy guard (see docstring above): None, or a reference to
# whichever CE2Session currently owns the one process-wide Kineto profiler.
# Training is single-threaded/single-process here, so a plain global is
# sufficient (no lock needed).
_ACTIVE_SESSION = None

# Name of the record_function marker a NESTED ce2_boundary() call tags its
# region with, inside the active session's own profiler. Not a "region
# namespace" the way ce_profiler.py's REGION_PREFIX is -- CE2 only ever splits
# a window into "boundary" vs "everything else", so one fixed tag suffices;
# multiple ce2_boundary() calls within one active session (e.g. InfLoRA's two
# DualGPM passes) each emit this same tag and are summed together by
# _split_totals, matching CE2Accumulator's own "sum multiple sub-calls into
# one bucket" semantics.
BOUNDARY_TAG = "ce2/boundary"


def _event_name(evt):
    name = getattr(evt, "key", None)
    if not isinstance(name, str):
        name = getattr(evt, "name", "")
    return name if isinstance(name, str) else ""


def _split_totals(prof):
    """Partition a finished profile's self-time/self-macs into (train_totals,
    boundary_totals), where "boundary" is everything that happened inside a
    BOUNDARY_TAG record_function scope (however deep/however many separate
    occurrences) and "train" is everything else. See module docstring for why
    summing self_* fields over every event, tagged or not, is double-count-safe
    regardless of tree shape. Returns two dicts, each {macs, device_seconds,
    host_seconds}. If the profile has no BOUNDARY_TAG events at all,
    boundary_totals comes back all-zero and train_totals equals the whole
    window's total -- i.e. this degrades to the old (pre-split) behavior
    exactly when nothing was tagged."""
    try:
        events = prof.events()
    except Exception as e:                                    # pragma: no cover
        logging.warning("[CE2 profiler] could not read profiler events: %s", e)
        zero = dict(macs=0.0, device_seconds=0.0, host_seconds=0.0)
        return dict(zero), dict(zero)

    boundary_ids = set()

    def _mark_subtree(evt):
        boundary_ids.add(id(evt))
        for child in (getattr(evt, "cpu_children", None) or ()):
            _mark_subtree(child)

    for evt in events:
        if _event_name(evt) == BOUNDARY_TAG:
            _mark_subtree(evt)

    buckets = {"train": {"macs": 0.0, "device_us": 0.0, "host_us": 0.0},
               "boundary": {"macs": 0.0, "device_us": 0.0, "host_us": 0.0}}
    for evt in events:
        bucket = buckets["boundary"] if id(evt) in boundary_ids else buckets["train"]
        flops = getattr(evt, "flops", None)
        if flops:
            bucket["macs"] += float(flops) / 2.0   # FLOPs -> MACs, same convention as ops_ledger.py
        self_device = getattr(evt, "self_cuda_time_total", None)
        if self_device is None:
            self_device = getattr(evt, "self_device_time_total", 0.0)
        bucket["device_us"] += float(self_device or 0.0)
        bucket["host_us"] += float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)

    def _finish(b):
        return dict(macs=b["macs"], device_seconds=b["device_us"] / 1e6, host_seconds=b["host_us"] / 1e6)

    return _finish(buckets["train"]), _finish(buckets["boundary"])


class CE2Session:
    """One coarse profiling window: wall-clock + torch.profiler(with_flops=True)
    together. `macs`/`device_seconds`/`host_seconds` cover the WHOLE window
    (unchanged meaning from before); `boundary_macs`/`boundary_device_seconds`/
    `boundary_host_seconds` expose just the portion tagged BOUNDARY_TAG by a
    NESTED ce2_boundary() call, if any occurred during this window (see module
    docstring) -- zero if none did. `ok=False` (all-zero totals, wall_seconds
    still valid) if the profiler failed to start/harvest -- never raises,
    mirroring CEProfileSession's "a broken measurement must not kill a
    multi-hour run" convention."""

    def __init__(self, device):
        self.device = device
        self.wall_seconds = 0.0
        self.macs = 0.0
        self.device_seconds = 0.0
        self.host_seconds = 0.0
        self.boundary_macs = 0.0
        self.boundary_device_seconds = 0.0
        self.boundary_host_seconds = 0.0
        self.ok = False
        self._prof = None
        self._t0 = None

    def __enter__(self):
        global _ACTIVE_SESSION
        self._t0 = time.time()
        if _ACTIVE_SESSION is not None:
            # Another CE2Session (always the enclosing per-task "total"
            # window -- see module docstring) already owns the one
            # process-wide Kineto profiler. Opening a second one here would
            # silently steal/stop it, corrupting the OUTER session's harvest
            # at its own, later __exit__. Fall back to wall-clock-only for
            # this (nested) session instead -- callers that want compute
            # attribution while nested should use ce2_boundary(), which tags
            # a region in the ACTIVE session rather than opening a CE2Session
            # directly.
            self._prof = None
            return self
        activities = [torch.profiler.ProfilerActivity.CPU]
        if getattr(self.device, "type", None) == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            self._prof = torch.profiler.profile(activities=activities, with_flops=True)
            self._prof.__enter__()
            _ACTIVE_SESSION = self
        except Exception as e:                                # pragma: no cover
            logging.warning("[CE2 profiler] failed to start: %s", e)
            self._prof = None
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_SESSION
        if self._prof is not None:
            try:
                if getattr(self.device, "type", None) == "cuda":
                    torch.cuda.synchronize()
                self._prof.__exit__(exc_type, exc, tb)
                if exc_type is None:
                    train_totals, boundary_totals = _split_totals(self._prof)
                    self.macs = train_totals["macs"] + boundary_totals["macs"]
                    self.device_seconds = train_totals["device_seconds"] + boundary_totals["device_seconds"]
                    self.host_seconds = train_totals["host_seconds"] + boundary_totals["host_seconds"]
                    self.boundary_macs = boundary_totals["macs"]
                    self.boundary_device_seconds = boundary_totals["device_seconds"]
                    self.boundary_host_seconds = boundary_totals["host_seconds"]
                    self.ok = True
            except Exception as e:                            # pragma: no cover
                logging.warning("[CE2 profiler] harvest failed: %s", e)
            finally:
                self._prof = None
                _ACTIVE_SESSION = None
        self.wall_seconds = time.time() - self._t0
        return False

    def totals(self):
        return dict(wall_seconds=self.wall_seconds, macs=self.macs,
                     device_seconds=self.device_seconds, host_seconds=self.host_seconds)

    def boundary_totals(self):
        """Just the BOUNDARY_TAG-tagged portion of this window, if any --
        {macs, device_seconds, host_seconds}. wall_seconds isn't included here:
        a nested ce2_boundary() call already measures its own wall-clock time
        directly (see ce2_boundary()), so there's no separate wall figure to
        recover from the split."""
        return dict(macs=self.boundary_macs, device_seconds=self.boundary_device_seconds,
                     host_seconds=self.boundary_host_seconds)


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

    def add_wall_only(self, wall_seconds):
        """Used by ce2_boundary()'s nested branch: only wall-clock is known
        synchronously (the compute split comes later, from the active
        session's own harvest -- see merge_split() below)."""
        self._totals["wall_seconds"] += wall_seconds

    def merge_split(self, split_totals):
        """Fold in a CE2Session.boundary_totals() dict (macs/device_seconds/
        host_seconds only) after the fact -- called by trainer.py once the
        outer "total" session has closed and its tag-split harvest is
        available, for whatever nested ce2_boundary() call(s) fired during
        that window (see module docstring)."""
        self._totals["macs"] += split_totals.get("macs", 0.0)
        self._totals["device_seconds"] += split_totals.get("device_seconds", 0.0)
        self._totals["host_seconds"] += split_totals.get("host_seconds", 0.0)

    def totals(self):
        return dict(self._totals)


@contextmanager
def ce2_boundary(model):
    """Method-side entry point: `with ce2_boundary(self): self._compress()`
    (or whatever the method's own boundary op is). Plain passthrough (no
    profiler, zero overhead) whenever model._ce2_boundary_acc is unset -- i.e.
    a complete no-op on every run that doesn't set ce2_enabled -- so it is
    always safe to leave these wraps in method code permanently.

    Two modes depending on whether another CE2Session is already active
    (always the enclosing per-task "total" window, when this fires from inside
    incremental_train()):
      - Not nested (e.g. a call from after_task(), which runs after the total
        session has already closed): opens its own real CE2Session as before,
        full macs/device/host measurement.
      - Nested: cannot open a second torch.profiler (see module docstring), so
        it tags this region with record_function(BOUNDARY_TAG) inside the
        ACTIVE session's own profiler instead, and records wall-clock time
        directly. The compute totals for this call aren't known yet at this
        point -- they arrive later, when the active session's own __exit__
        harvests and splits its profile; trainer.py is responsible for folding
        that split back into this task's boundary accumulator via
        merge_split() once the active (total) session closes.
    """
    acc = getattr(model, "_ce2_boundary_acc", None)
    if acc is None:
        yield
        return
    if _ACTIVE_SESSION is not None:
        t0 = time.time()
        with torch.profiler.record_function(BOUNDARY_TAG):
            yield
        acc.add_wall_only(time.time() - t0)
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
