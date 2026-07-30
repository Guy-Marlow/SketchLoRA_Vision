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
                     aux_macs_per_step=0.0, boundary_macs=None, auxiliary_pass_macs=None):
        """One record per cycle/task.

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
        ops_fb = steps_per_epoch * (step_macs_fwd + step_macs_bwd)
        total_steps = n_epochs * steps_per_epoch
        ops_total = (total_steps * (step_macs_fwd + step_macs_bwd + aux_macs_per_step)
                     + sum((boundary_macs or {}).values())
                     + sum((auxiliary_pass_macs or {}).values()))
        rec["ops_fb"] = ops_fb
        rec["ops_total"] = ops_total
        self.records.append(rec)
        with open(self.out_path, "w") as f:
            json.dump(self.records, f, indent=2)
        return rec


def compute_ce(ledger_path_or_records, eps=20):
    """CE = min(1, (1/N) * sum_i [Ops_fb(Tr_i) * eps / Ops(Tr_i)]), computed
    OFFLINE from a saved ledger (path) or an in-memory records list."""
    if isinstance(ledger_path_or_records, str):
        records = json.load(open(ledger_path_or_records))
    else:
        records = ledger_path_or_records
    if not records:
        return None
    n = len(records)
    total = 0.0
    for rec in records:
        if rec["ops_total"] <= 0:
            continue
        total += rec["ops_fb"] * eps / rec["ops_total"]
    return min(1.0, total / n)
