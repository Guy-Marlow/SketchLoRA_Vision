"""Computational-Efficiency (CE) metric infrastructure (impl_plan_7.27.2026 Part
2), after Diaz-Rodriguez, Lomonaco, Filliat & Maltoni 2018.

CE = min(1, (1/N) * sum_i [ Ops_fb(Tr_i) * eps / Ops(Tr_i) ])

  Ops_fb(Tr_i): MACs for ONE forward + ONE backward pass over Tr_i, measured
                (torch.profiler, with_flops=True), never assumed.
  Ops(Tr_i):    TOTAL MACs actually spent learning Tr_i (all epochs + every
                auxiliary mechanism: penalties, hooks, boundary ops, extra
                passes).
  eps:          set to E=20 (the shared epoch budget) everywhere in this
                campaign -- a method with zero overhead beyond the matched
                budget scores exactly 1.0.
  Unit Tr_i:    oracle runs -> real task i, N = task count. bounded-memory
                runs -> cycle i, N = cycle count (cycles are the learning
                exposures under boundary-agnostic streaming).

Counting convention: MACs (mul-adds) everywhere. FLOPs = 2x MACs -- this
module never mixes the two; torch.profiler reports FLOPs, divided by 2 the
moment they're read.

Per-run artifact: ops_ledger.json, one record per cycle/task:
  {"unit_idx": i, "n_steps": ..., "step_macs_fwd": ..., "step_macs_bwd": ...,
   "aux_macs_per_step": ..., "boundary_macs": {...itemized by category...},
   "auxiliary_pass_macs": {...itemized...}}
CE is computed OFFLINE from this ledger (persist-everything, matching the
rest of this project's metrics conventions), never inline during training.
"""
import json
import os

import torch


# ---- Ops_fb measurement (profiler-based, never assumed) -------------------

def measure_step_macs(train_step_fwd_only, train_step_fwd_bwd, device):
    """Measure one training step's forward-only and forward+backward MACs via
    torch.profiler(with_flops=True), two separate profiled calls (avoids
    needing to disambiguate fwd/bwd at the op-record level within one trace).
    Returns (fwd_macs, bwd_macs). Callers must not update running-stat /
    optimizer state as a side effect they care about -- these are throwaway
    profiling calls on a real batch, run once per (method, checkpoint config).

    train_step_fwd_only()   : calls the model's forward pass only, returns nothing
    train_step_fwd_bwd()    : calls forward + loss.backward() (no optimizer.step()
                               required -- MACs don't depend on the step itself)
    """
    def _profiled_flops(fn):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
        ) as prof:
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        return total_flops / 2.0   # FLOPs -> MACs

    fwd_macs = _profiled_flops(train_step_fwd_only)
    total_macs = _profiled_flops(train_step_fwd_bwd)
    bwd_macs = max(0.0, total_macs - fwd_macs)
    return fwd_macs, bwd_macs


def fvcore_cross_check(model, sample_input):
    """Cross-check forward MACs against fvcore, within 5% (impl_plan_7.27.2026
    sec 2.2). fvcore is NOT installed in this environment (checked: `import
    fvcore` -> ModuleNotFoundError) -- flagged rather than silently skipped.
    Returns None (cross-check not performed) if fvcore is unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None
    flops = FlopCountAnalysis(model, sample_input)
    return flops.total() / 2.0   # FLOPs -> MACs


# ---- ledger -----------------------------------------------------------------

class OpsLedger:
    """Per-run artifact, one record per learning-exposure unit (cycle under
    bounded-memory streaming, task under the oracle path). Written to disk on
    every record (same "persist as you go" convention as utils/metrics_logger.py),
    so a killed run still leaves a readable partial ledger."""

    def __init__(self, out_dir, tag):
        os.makedirs(out_dir, exist_ok=True)
        self.out_path = os.path.join(out_dir, "ops_ledger_{}.json".format(tag))
        self.records = []

    def record_unit(self, unit_idx, steps_per_epoch, n_epochs, step_macs_fwd, step_macs_bwd,
                     aux_macs_per_step=0.0, boundary_macs=None, auxiliary_pass_macs=None,
                     measured_step_regions=None, measured_boundary_regions=None,
                     baseline_step_macs_fwd=None, baseline_step_macs_bwd=None,
                     profile_provenance=None, nearest_latent_task=None):
        """One record per cycle/task.

        The `measured_*` / `baseline_*` / `profile_provenance` arguments are the
        2026-08-03 measured-CE extension (docs/ce_profiling_implementation_plan.md,
        utils/ce_profiler.py). They are OPTIONAL and additive: when omitted this
        method behaves exactly as it always did, and every previously-written
        ledger stays readable by compute_ce() unchanged. When supplied, the
        record carries BOTH the analytic-formula numbers (aux_macs_per_step /
        boundary_macs, as before) and the profiler-measured ones side by side --
        deliberately, so the two can be compared directly on the same cycles.
        That comparison IS the validation criterion in the plan's section 7; it
        is not redundancy to be cleaned up until the measured path has been
        confirmed against a live run.

        measured_step_regions / measured_boundary_regions: {label: {"macs",
          "device_seconds", "host_seconds", "sync_ops", "n_calls"}} from
          utils/ce_profiler.py. Step regions are already normalised to PER-STEP.
        baseline_step_macs_fwd/bwd: the shared SeqLoRA-equivalent (single-slot,
          merge=False) step cost -- plan convention R2. Using this as the CE
          numerator instead of each method's own forward stops a method's extra
          forward cost from cancelling between numerator and denominator.
        nearest_latent_task: OPTIONAL, added 2026-08-03 for the CE smoke test
          (docs/ce_profiling_implementation_plan.md request: "data at each task
          boundary, so we can see how each method's compute is beginning to
          scale with task count"). Int, same np.searchsorted(task_image_cumends,
          cum_images) computation models/bounded_memory_mixin.py's own accuracy
          eval already uses to label its "_nearest_latent_task" field -- write-only
          telemetry, exactly like that field (see this module's own leak-audit
          convention): it lets downstream analysis find "which cycles cover real
          task N" directly from the CE ledger alone, without re-deriving the
          image-count-to-task mapping via a separate DataManager pass. Does not
          affect ops_fb/ops_total or any CE computation.
        profile_provenance: {kind: {"profiled": bool, "held_from_cycle": int|None}}
          -- one entry per controller kind actually in use (the driver passes
          "step"/"boundary_begin"/"boundary_end"), each saying whether THIS
          cycle's measured numbers for that kind were freshly measured or held
          from the last profiled cycle, so downstream analysis never mistakes a
          held value for a measured one. compute_ce_report() also accepts a
          flat {"profiled": bool, ...} for back-compat, but the driver's own
          call site always passes the nested, per-kind form.

        Ops_fb(Tr_i) = ONE epoch's worth of fwd+bwd = steps_per_epoch * (fwd+bwd).
        Ops(Tr_i)    = ALL n_epochs' worth (+ any per-step auxiliary cost,
                       charged every step across every epoch) + one-off
                       boundary/auxiliary-pass costs for this unit.

        This is the definition that makes a zero-overhead method (SeqLoRA:
        aux=0, boundary=0, aux_pass=0) score CE_i = eps/n_epochs exactly --
        with eps=n_epochs=20 (this campaign's shared epoch budget), CE_i=1.0.

        `aux_macs_per_step` (e.g. O-LoRA's per-step orthogonality penalty,
        TreeLoRA's per-step regularizer) is charged n_epochs*steps_per_epoch
        times (every step, every epoch); `boundary_macs`/`auxiliary_pass_macs`
        are itemized dicts of one-off per-unit costs (folds, DualGPM SVDs, CA
        alignment events, InfLoRA's extra covariance passes, ...) charged once
        per unit regardless of epoch count."""
        rec = {
            "unit_idx": unit_idx,
            "steps_per_epoch": steps_per_epoch,
            "n_epochs": n_epochs,
            "step_macs_fwd": step_macs_fwd,
            "step_macs_bwd": step_macs_bwd,
            "aux_macs_per_step": aux_macs_per_step,
            "boundary_macs": boundary_macs or {},
            "auxiliary_pass_macs": auxiliary_pass_macs or {},
        }
        if nearest_latent_task is not None:
            rec["nearest_latent_task"] = nearest_latent_task
        ops_fb = steps_per_epoch * (step_macs_fwd + step_macs_bwd)
        total_steps = n_epochs * steps_per_epoch
        ops_total = (total_steps * (step_macs_fwd + step_macs_bwd + aux_macs_per_step)
                     + sum((boundary_macs or {}).values())
                     + sum((auxiliary_pass_macs or {}).values()))
        rec["ops_fb"] = ops_fb
        rec["ops_total"] = ops_total

        # ---- measured-CE extension (2026-08-03) ----
        if measured_step_regions is not None or measured_boundary_regions is not None:
            from utils.ce_profiler import charged_macs, charged_seconds
            rec["measured_step_regions"] = measured_step_regions or {}
            rec["measured_boundary_regions"] = measured_boundary_regions or {}
            rec["profile_provenance"] = profile_provenance or {}
            # charged_macs excludes "_excluded/..." labels -- our own diagnostic
            # apparatus must never be charged to the method (plan convention R3).
            aux_measured = charged_macs(measured_step_regions)
            boundary_measured = charged_macs(measured_boundary_regions)
            rec["aux_macs_per_step_measured"] = aux_measured
            rec["boundary_macs_measured"] = boundary_measured
            rec["aux_seconds_per_step_measured"] = charged_seconds(measured_step_regions)
            rec["boundary_seconds_measured"] = charged_seconds(measured_boundary_regions)
            rec["aux_sync_ops_per_step"] = sum(
                v.get("sync_ops", 0) for k, v in (measured_step_regions or {}).items()
                if not k.startswith("_excluded/"))
            rec["boundary_sync_ops"] = sum(
                v.get("sync_ops", 0) for k, v in (measured_boundary_regions or {}).items()
                if not k.startswith("_excluded/"))
            rec["ops_total_measured"] = (
                total_steps * (step_macs_fwd + step_macs_bwd + aux_measured)
                + boundary_measured)
        if baseline_step_macs_fwd is not None:
            # Plan convention R2: the shared, method-independent numerator. The
            # method's OWN forward excess over this baseline stays in ops_total,
            # where it is charged instead of cancelling.
            rec["baseline_step_macs_fwd"] = baseline_step_macs_fwd
            rec["baseline_step_macs_bwd"] = baseline_step_macs_bwd
            rec["ops_fb_baseline"] = steps_per_epoch * (
                baseline_step_macs_fwd + baseline_step_macs_bwd)
            rec["merged_forward_excess_per_step"] = (
                (step_macs_fwd + step_macs_bwd)
                - (baseline_step_macs_fwd + baseline_step_macs_bwd))

        self.records.append(rec)
        with open(self.out_path, "w") as f:
            json.dump(self.records, f, indent=2)
        return rec


def compute_ce(ledger_path_or_records, eps=20, source="formula", baseline_numerator=False):
    """CE = min(1, (1/N) * sum_i [Ops_fb(Tr_i) * eps / Ops(Tr_i)]), computed
    OFFLINE from a saved ledger (path) or an in-memory records list.

    source: "formula" (default, the original analytic ce_formulas.py path -- and
      the ONLY option for every ledger written before 2026-08-03) or "measured"
      (profiler-measured regions, utils/ce_profiler.py). "measured" falls back to
      the formula figure for any record lacking measured data, so a partially
      instrumented run still computes; the returned dict from compute_ce_report()
      says how many records actually had measurements.
    baseline_numerator: plan convention R2. False keeps the original behaviour
      (each method's own forward in the numerator, where its cost cancels against
      the denominator). True uses the shared SeqLoRA-equivalent baseline, so a
      method's extra forward cost is charged rather than cancelled. Only
      available on ledgers written with baseline_step_macs_fwd recorded.
    """
    if isinstance(ledger_path_or_records, str):
        records = json.load(open(ledger_path_or_records))
    else:
        records = ledger_path_or_records
    if not records:
        return None
    n = len(records)
    total = 0.0
    for rec in records:
        ops_total = rec["ops_total"]
        if source == "measured":
            ops_total = rec.get("ops_total_measured", ops_total)
        if ops_total <= 0:
            continue
        ops_fb = rec["ops_fb"]
        if baseline_numerator:
            ops_fb = rec.get("ops_fb_baseline", ops_fb)
        total += ops_fb * eps / ops_total
    return min(1.0, total / n)


def compute_ce_report(ledger_path_or_records, eps=20):
    """Every CE variant the ledger supports, plus provenance -- so a report can
    state plainly how many cycles were actually profiled rather than implying the
    whole run was measured. Returns None for variants the ledger can't support."""
    if isinstance(ledger_path_or_records, str):
        records = json.load(open(ledger_path_or_records))
    else:
        records = ledger_path_or_records
    if not records:
        return None
    n_measured = sum(1 for r in records if "ops_total_measured" in r)
    # *** UNTESTED as of 2026-08-03 *** -- FIXED (found by a synthetic,
    # no-GPU-needed unit test of this exact function, run while implementing
    # Step 5): profile_provenance is a dict OF per-kind provenance dicts
    # (models/bounded_memory_mixin.py passes {"step": {...},
    # "boundary_begin": {...}, "boundary_end": {...}}), NOT the flat
    # {"profiled": bool, ...} this function originally assumed -- the original
    # `.get("profiled")` on the outer dict always returned None (no such key at
    # that level), so n_actually_profiled silently read 0 on every ledger ever
    # written, regardless of how many cycles were really profiled. A record now
    # counts as profiled if ANY of its tracked kinds were profiled that cycle.
    def _any_kind_profiled(rec):
        prov = rec.get("profile_provenance") or {}
        if "profiled" in prov:
            return bool(prov["profiled"])   # back-compat: a flat provenance dict
        return any(isinstance(v, dict) and v.get("profiled") for v in prov.values())
    n_profiled = sum(1 for r in records if _any_kind_profiled(r))
    has_baseline = any("ops_fb_baseline" in r for r in records)
    out = {
        "n_records": len(records),
        "n_with_measured": n_measured,
        "n_actually_profiled": n_profiled,
        "ce_formula": compute_ce(records, eps=eps, source="formula"),
        "ce_measured": compute_ce(records, eps=eps, source="measured") if n_measured else None,
        "ce_formula_baseline_numerator": (
            compute_ce(records, eps=eps, source="formula", baseline_numerator=True)
            if has_baseline else None),
        "ce_measured_baseline_numerator": (
            compute_ce(records, eps=eps, source="measured", baseline_numerator=True)
            if (n_measured and has_baseline) else None),
    }
    # *** UNTESTED as of 2026-08-03 *** -- ADDED (user-flagged: logging
    # ce_formula as THE headline number is unacceptable in oracle mode, where
    # the formula hooks are skipped entirely and ce_formula reads ~1.0 for
    # every method by construction -- that is not "the CE," it's an inert
    # placeholder). ce_best resolves to the single most trustworthy number
    # available, in preference order: measured+baseline-numerator (both fixes
    # applied) > measured alone > formula+baseline-numerator > formula alone.
    # This is what callers should log/print as "the" CE value; the rest of
    # this dict stays available for anyone who wants the full breakdown.
    out["ce_best"] = next(
        (out[k] for k in ("ce_measured_baseline_numerator", "ce_measured",
                          "ce_formula_baseline_numerator", "ce_formula")
         if out[k] is not None), None)
    out["ce_best_source"] = next(
        (k for k in ("ce_measured_baseline_numerator", "ce_measured",
                     "ce_formula_baseline_numerator", "ce_formula")
         if out[k] is not None), None)
    return out
