"""Lazy-merge bolt-on for SketchLoRA (impl_plan_7.27.2026 sec 1.2): delay
folding the residual into the sketch past every single chunk boundary.

Two modes, both boundary-blind (no data volume/cycle-count/real-task info
read beyond the residual's own trained weights):

  period:  generalizes the EXISTING `svd_period` (P) hyperparameter (models/
           sketchlora.py's fixed-target-rank sensitivity sweep, previously
           oracle/task-boundary only) to the streaming/bounded-memory path.
           Not a separate counter -- `lazy_merge_period` just sets
           `self.svd_period`, and folding fires at period boundaries exactly
           like the oracle path's `at_period_boundary` check.
  plateau: NEW. Tracks drift d_c = ||R_c - R_{c-1}||_F / (||R_{c-1}||_F + eps)
           of the (single) residual's own factor product across cycles.
           Folds once d_c falls below `lazy_merge_delta` for 2 CONSECUTIVE
           cycles, or once `lazy_merge_max_holdoff` cycles have passed since
           the last fold (staleness cap), whichever comes first.

Both leave the OFF path (svd_period=1, mode="off") exactly as it always was:
_stream_slot()'s period-aware routing reduces to the single fixed RESIDUAL
slot when svd_period==1, and PlateauTracker is never constructed/consulted
when mode != "plateau".
"""
import torch


class PlateauTracker:
    """Per-Learner drift-plateau fold trigger (impl_plan_7.27.2026 sec 1.2,
    mode=plateau). Call `should_fold(residual_products)` once per stream
    cycle, after that cycle's training, with the CURRENT cycle's residual
    factor products (one [d, d] tensor per tracked LoRA module, in a fixed
    consistent order across calls). Returns True exactly when a fold should
    fire this cycle; the caller (sketchlora.py) is responsible for actually
    folding and then calling `reset()` to clear post-fold state.
    """

    def __init__(self, delta, max_holdoff, eps=1e-8):
        self.delta = delta
        self.max_holdoff = max_holdoff
        self.eps = eps
        self._prev = None            # list of [d,d] tensors, or None before cycle 0
        self._below_streak = 0       # consecutive cycles with d_c < delta
        self._cycles_since_fold = 0
        self.last_drift = None       # for logging
        self.last_trigger_reason = None

    def reset(self):
        """Call right after a fold: the residual is now zero, so the next
        cycle's drift is measured against a clean zero baseline."""
        self._prev = None
        self._below_streak = 0
        self._cycles_since_fold = 0

    @torch.no_grad()
    def should_fold(self, residual_products):
        self._cycles_since_fold += 1
        if self._cycles_since_fold >= self.max_holdoff:
            self.last_trigger_reason = "max_holdoff"
            self.last_drift = self._compute_mean_drift(residual_products)
            return True

        d_c = self._compute_mean_drift(residual_products)
        self.last_drift = d_c
        self._prev = [r.detach().clone() for r in residual_products]

        if d_c < self.delta:
            self._below_streak += 1
        else:
            self._below_streak = 0

        if self._below_streak >= 2:
            self.last_trigger_reason = "plateau"
            return True
        self.last_trigger_reason = None
        return False

    def _compute_mean_drift(self, residual_products):
        if self._prev is None:
            # Cycle 0 (or right after a reset): no previous residual to compare
            # against -- the residual started this cycle at exactly zero (a
            # fresh kaiming-A/zero-B reset), so drift is ||R_c||_F / eps,
            # deliberately large/undefined-large -- a fold cannot plateau-
            # trigger on the very first cycle after a reset, which is the
            # correct behavior (there is nothing to plateau FROM yet).
            return float("inf")
        diffs = []
        for r_c, r_prev in zip(residual_products, self._prev):
            num = (r_c - r_prev).float().norm()
            den = r_prev.float().norm() + self.eps
            diffs.append((num / den).item())
        return sum(diffs) / len(diffs) if diffs else float("inf")
