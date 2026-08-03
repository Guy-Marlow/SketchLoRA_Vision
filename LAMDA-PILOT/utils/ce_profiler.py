"""Measured-CE region profiling (docs/ce_profiling_implementation_plan.md).

*** UNTESTED as of 2026-08-03 *** -- written while local GPUs were unavailable
(thermal/damage risk). Verified by static reasoning and API-signature checks
against torch 2.4.1 only; NO live run has exercised any of this. Every
consumer of this module inherits that caveat. See the plan doc's section 8.

WHY THIS EXISTS
---------------
utils/ce_formulas.py estimates each method's auxiliary cost from hand-derived
analytic formulas. Three separate defects were found in those formulas on
2026-08-03 by manual code re-reading -- none by measurement, because nothing
measured. This module replaces estimation with direct torch.profiler
measurement of named regions inside each method's own code.

DESIGN
------
1. Methods tag their overhead regions with `with ce_region("olora/orth_penalty"):`.
   When no profiling session is active (the common case -- ~96% of cycles under
   the default sampling cadence) that call returns a shared no-op singleton and
   costs one module-global identity check. It is safe to leave these tags in
   permanently.

2. The driver (models/bounded_memory_mixin.py) opens a CEProfileSession around a
   SAMPLE of training steps and around the boundary call. Two separate sessions
   rather than one whole-cycle session: it bounds trace size, and it maps
   exactly onto the ledger's own two cost categories (per-step vs per-boundary).

3. Regions are attributed EXCLUSIVELY: _walk_exclusive stops descending when it
   meets a nested ce/ scope, so an inner region's cost is never also counted in
   its enclosing region. Nesting is therefore safe (unlike a naive sum).

4. Both MACs and time are recorded, per convention R5 of the plan: MACs cannot
   see GPU->CPU syncs, Python-loop kernel-launch overhead, or bandwidth-bound
   copies, which together are most of TreeLoRA's real cost. `sync_ops` counts
   device->host synchronisation points explicitly, since those are the specific
   thing a MAC ledger is blind to.

TIME SEMANTICS (read before trusting the seconds fields)
--------------------------------------------------------
CUDA is asynchronous, so there is no single honest "wall clock" for a region
without inserting synchronisation that itself perturbs what is measured. Two
different numbers are therefore reported and NEITHER should be silently treated
as "the" time:
  device_seconds: summed self CUDA time of the ops inside the region -- the
                  device actually did this much work. Does not include host
                  stalls, and does not account for kernels overlapping with
                  unrelated queued work.
  host_seconds:   the region scope's own CPU-side elapsed time -- includes
                  Python-loop overhead, kernel-launch cost, and any stall on a
                  device->host sync. For a launch-bound or sync-bound region
                  (TreeLoRA's tree build) this is the meaningful one; for a
                  compute-bound region it understates real device cost.
"""

import collections
import logging

import torch

REGION_PREFIX = "ce/"

# Ops that force a device->host synchronisation. A region whose cost is mostly
# these is invisible to MAC accounting entirely (plan convention R5) -- counted
# separately so that fact is legible in the ledger rather than inferred.
_SYNC_OP_MARKERS = (
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::is_nonzero",
    "Memcpy DtoH",
    "cudaMemcpyAsync",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
)

# Set by CEProfileSession.__enter__/__exit__. Module-global rather than passed
# through every call site because ce_region() is invoked from deep inside method
# code that has no reason to thread a profiler handle around.
_ACTIVE_SESSION = None


class _NullRegion:
    """Zero-allocation no-op context. Returned by ce_region() when no session is
    active, so leaving tags in production code costs one identity check."""
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL_REGION = _NullRegion()


def ce_region(label):
    """Tag a CE overhead region. No-op unless a CEProfileSession is active.

    label: "<method>/<region>", e.g. "sketchlora/fold_merge_randsvd". Use the
    "_excluded/" method prefix for work that belongs to OUR measurement
    apparatus rather than the method's algorithm (plan convention R3) -- those
    are harvested and reported like any other region but must never be summed
    into a method's charged overhead.
    """
    if _ACTIVE_SESSION is None:
        return _NULL_REGION
    return torch.profiler.record_function(REGION_PREFIX + label)


def ce_profiling_active():
    """True while a session is open. For code that wants to skip work that only
    exists to support profiling; NOT needed to guard ce_region() itself."""
    return _ACTIVE_SESSION is not None


def _event_name(evt):
    name = getattr(evt, "key", None)
    if not isinstance(name, str):
        name = getattr(evt, "name", "")
    return name if isinstance(name, str) else ""


def _walk_exclusive(evt, acc):
    """Accumulate flops/time/sync-count over evt's subtree, STOPPING at any
    nested ce/ scope (that scope harvests its own subtree, so descending into it
    here would double-count -- see design note 3)."""
    flops = getattr(evt, "flops", None)
    if flops:
        acc["flops"] += float(flops)
    self_cuda = getattr(evt, "self_cuda_time_total", None)
    if self_cuda is None:
        # torch renamed this to self_device_time_total in later versions; 2.4.1
        # still has the cuda-named attribute, but don't assume.
        self_cuda = getattr(evt, "self_device_time_total", 0.0)
    acc["device_us"] += float(self_cuda or 0.0)
    name = _event_name(evt)
    if any(marker in name for marker in _SYNC_OP_MARKERS):
        acc["sync_ops"] += 1
    for child in (getattr(evt, "cpu_children", None) or ()):
        if _event_name(child).startswith(REGION_PREFIX):
            continue   # nested region -- harvested separately, exclusive attribution
        _walk_exclusive(child, acc)


def _harvest(prof):
    """Read per-region MACs/time/sync-counts out of a finished profile.

    torch.profiler reports FLOPs; this project's ledger convention is MACs
    everywhere (utils/ops_ledger.py's module docstring), so divide by 2 at the
    boundary, exactly as measure_step_macs already does.
    """
    try:
        events = prof.events()
    except Exception as e:                                   # pragma: no cover
        logging.warning("[CE profiler] could not read profiler events: %s", e)
        return {}
    out = {}
    for evt in events:
        name = _event_name(evt)
        if not name.startswith(REGION_PREFIX):
            continue
        label = name[len(REGION_PREFIX):]
        acc = {"flops": 0.0, "device_us": 0.0, "sync_ops": 0}
        _walk_exclusive(evt, acc)
        host_us = getattr(evt, "cpu_time_total", 0.0) or 0.0
        agg = out.setdefault(label, {"macs": 0.0, "device_seconds": 0.0,
                                      "host_seconds": 0.0, "sync_ops": 0, "n_calls": 0})
        agg["macs"] += acc["flops"] / 2.0
        agg["device_seconds"] += acc["device_us"] / 1e6
        agg["host_seconds"] += float(host_us) / 1e6
        agg["sync_ops"] += acc["sync_ops"]
        agg["n_calls"] += 1
    return out


class CEProfileSession:
    """One profiling window. Opens a torch.profiler, installs itself as the
    active session so ce_region() starts recording, and harvests per-region
    totals on exit.

    Never raises out of __exit__ on a harvest failure -- a broken measurement
    must not kill a multi-hour training run; it degrades to an empty region dict
    which the ledger records as "profiled but nothing captured", a state the
    downstream analysis can see and flag."""

    def __init__(self, device, kind="step"):
        self.device = device
        self.kind = kind
        self.regions = {}
        self.ok = False
        self._prof = None

    def __enter__(self):
        global _ACTIVE_SESSION
        if _ACTIVE_SESSION is not None:
            # Re-entrancy would make attribution ambiguous. Degrade to a no-op
            # window rather than corrupting the outer session's numbers.
            logging.warning("[CE profiler] nested session (%s) ignored", self.kind)
            return self
        activities = [torch.profiler.ProfilerActivity.CPU]
        if getattr(self.device, "type", None) == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            self._prof = torch.profiler.profile(activities=activities, with_flops=True)
            self._prof.__enter__()
        except Exception as e:                               # pragma: no cover
            logging.warning("[CE profiler] failed to start (%s): %s", self.kind, e)
            self._prof = None
            return self
        _ACTIVE_SESSION = self
        return self

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_SESSION
        if _ACTIVE_SESSION is self:
            _ACTIVE_SESSION = None
        if self._prof is None:
            return False
        try:
            if getattr(self.device, "type", None) == "cuda":
                # Required for correct device-time attribution: without it the
                # profiler's window can close before queued kernels complete.
                torch.cuda.synchronize()
            self._prof.__exit__(exc_type, exc, tb)
            if exc_type is None:
                self.regions = _harvest(self._prof)
                self.ok = True
        except Exception as e:                               # pragma: no cover
            logging.warning("[CE profiler] harvest failed (%s): %s", self.kind, e)
        finally:
            self._prof = None
        return False


class CEProfileController:
    """Sampling schedule + hold-between-samples bookkeeping (plan section 3.2).

    Profiling every cycle would perturb the run it measures; profiling once
    would miss precisely the growth trends (O-LoRA's slot count, InfLoRA's
    DualGPM basis, TreeLoRA's task_id, SketchLoRA's r_hat) that motivate the
    whole exercise. So: profile every `profile_every`-th cycle plus a few forced
    ones, hold the last measurement in between, and record in every ledger entry
    whether its numbers were measured or held -- so downstream analysis can
    interpolate honestly instead of silently treating held values as measured.
    """

    def __init__(self, device, profile_every=25, enabled=True, force_cycles=(0, 1)):
        self.device = device
        self.profile_every = max(1, int(profile_every))
        self.enabled = bool(enabled)
        self.force_cycles = set(force_cycles or ())
        # Any `kind` string works (not just "step"/"boundary") -- e.g. the
        # driver uses "boundary_begin"/"boundary_end" as two separately-tracked
        # kinds for _stream_begin_chunk vs _stream_end_chunk (2026-08-03 fix,
        # see docs/ce_profiling_implementation_plan.md sec 4.3). defaultdict
        # rather than a fixed set of keys so a new kind never KeyErrors.
        self._last = collections.defaultdict(dict)
        self._last_profiled_cycle = collections.defaultdict(lambda: None)
        self._cycle_idx = None
        self._profiling_this_cycle = False

    def begin_cycle(self, cycle_idx, is_final=False):
        self._cycle_idx = cycle_idx
        self._profiling_this_cycle = self.enabled and (
            is_final or cycle_idx in self.force_cycles
            or cycle_idx % self.profile_every == 0)
        return self._profiling_this_cycle

    @property
    def profiling_this_cycle(self):
        return self._profiling_this_cycle

    def session(self, kind):
        """A CEProfileSession if this cycle is being profiled, else a no-op
        context. Callers use `with controller.session("boundary"):` and then
        `controller.commit(session, "boundary")`."""
        if not self._profiling_this_cycle:
            return _NULL_REGION
        return CEProfileSession(self.device, kind=kind)

    def commit(self, session, kind, scale=1.0):
        """Record a just-finished session's regions as the held value for `kind`.

        scale: multiplier applied to MACs/time, used for per-step sessions that
        profiled K steps out of the cycle's total -- pass 1/K to normalise to
        per-step. Sync counts are scaled too (they are per-step counts, and the
        whole point of tracking them is per-step magnitude)."""
        if not isinstance(session, CEProfileSession) or not session.ok:
            return self._last[kind]
        scaled = {}
        for label, vals in session.regions.items():
            scaled[label] = {
                "macs": vals["macs"] * scale,
                "device_seconds": vals["device_seconds"] * scale,
                "host_seconds": vals["host_seconds"] * scale,
                "sync_ops": vals["sync_ops"] * scale,
                "n_calls": vals["n_calls"],
            }
        self._last[kind] = scaled
        self._last_profiled_cycle[kind] = self._cycle_idx
        return scaled

    def current(self, kind):
        """The value in force for this cycle -- freshly measured if this cycle
        was profiled, otherwise the last measured value, held."""
        return self._last[kind]

    def provenance(self, kind):
        return {
            "profiled": bool(self._profiling_this_cycle),
            "held_from_cycle": self._last_profiled_cycle[kind],
        }


# ---- charged-overhead reduction ------------------------------------------

def charged_macs(regions):
    """Sum a region dict's MACs, EXCLUDING "_excluded/..." labels.

    Plan convention R3: our own measurement apparatus (SketchLoRA's sketch_diag
    reconstruction, param hashing, MetricsLogger) runs inside the same code
    paths as the method's real work, and a method must never be charged for it.
    Those regions are still harvested and written to the ledger -- so the cost
    is visible and auditable -- but they never enter the CE numerator/denominator.
    """
    return sum(v["macs"] for k, v in (regions or {}).items()
               if not k.startswith("_excluded/"))


def charged_seconds(regions, which="device_seconds"):
    return sum(v[which] for k, v in (regions or {}).items()
               if not k.startswith("_excluded/"))


# ---- R2: shared baseline vs the method's own forward ----------------------

def measure_baseline_and_actual(network, inputs, targets, loss_fn, slot, merge, device):
    """Measure BOTH a SeqLoRA-equivalent baseline step and the method's own
    actual step (plan section 2, convention R2).

    The existing single measurement (utils/ops_ledger.py::measure_step_macs)
    profiles each method on its OWN routing, which puts every method's extra
    forward cost -- O-LoRA/InfLoRA/TreeLoRA's frozen_delta [d,d] matmul,
    SketchLoRA's 2*d*r_hat sketch slot -- into both the CE numerator (Ops_fb)
    and denominator (Ops_total), where it cancels. A method with an expensive
    forward and no auxiliary work consequently scores CE ~= 1.0 and appears
    free, which is wrong and is exactly the reason "fold the sketch into the
    backbone weights" looks like it does nothing under the current metric.

    Returning both lets the ledger use the SHARED baseline as Ops_fb (identical
    in kind for every method: one slot, no merge) while the method's real
    forward cost stays in Ops_total, where its excess is charged.

    Returns (baseline_fwd, baseline_bwd, actual_fwd, actual_bwd) in MACs.
    """
    from utils.ops_ledger import measure_step_macs

    def _mk(_merge):
        def _fwd_only():
            with torch.no_grad():
                network(inputs, task=slot, merge=_merge)

        def _fwd_bwd():
            out = network(inputs, task=slot, merge=_merge)
            loss_fn(out["logits"]).backward()
        return _fwd_only, _fwd_bwd

    # merge=False routes _lora_delta through the single-slot branch -- the same
    # work SeqLoRA does, on the same backbone, with the same batch. This is the
    # universal baseline; it is NOT method-specific.
    b_fwd_only, b_fwd_bwd = _mk(False)
    baseline = measure_step_macs(b_fwd_only, b_fwd_bwd, device)
    network.zero_grad()

    if not merge:
        # Method already runs the baseline routing (SeqLoRA, and any method
        # whose train_merge is False) -- no second measurement needed, and
        # measuring twice would only add profiler noise between two identical
        # configurations.
        actual = baseline
    else:
        a_fwd_only, a_fwd_bwd = _mk(True)
        actual = measure_step_macs(a_fwd_only, a_fwd_bwd, device)
        network.zero_grad()
    return baseline[0], baseline[1], actual[0], actual[1]
