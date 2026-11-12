"""KL-divergence drift detector for the new batch-incremental setting's
adapter-SPAWN trigger (boundary_mode to be determined -- see models/
stream_mixin.py-adjacent driver, not yet wired). Design specified by the
user via an external review (web-claude), not independently designed here
-- this file is a direct, careful transcription of that specification, not
a reinterpretation of it.

SIGNAL SOURCE: the frozen (no-adapter) backbone's own CLS-token feature,
recomputed fresh every step from the CURRENT batch's own (augmented)
inputs, under no_grad. No caching, no precomputed dataset-wide pass -- a
cache would leak future samples into a setting whose whole point is
streaming, and would use eval-transform inputs rather than what the
learner actually trains on. This file does not compute that feature itself
(the caller does, since it owns the backbone forward); it only consumes
the resulting [B, D] tensor.

DISTRIBUTION MODEL: diagonal (per-dimension) Gaussian over the D=768 CLS
feature. Two views are maintained:
  - RECENT: a true ring buffer over the last `window_batches` batches
    (default >=4 batches, sized so the pooled sample count is roughly 256).
    Its pooled mean/variance are recomputed from whatever is currently in
    the buffer.
  - REFERENCE: a running mean/variance updated via Welford's algorithm
    (parallel/group-merge form, since whole batches enter at once), fed
    ONLY by batches that have just aged OUT of the recent ring buffer --
    so the reference always lags the window and never includes what it is
    currently being compared against, and keeps absorbing the segment's
    own history for as long as the segment lasts rather than freezing at
    its first few batches.

KL is computed each step the reference has at least one observation:
    KL(recent || reference) = 0.5 * sum_d [
        var_recent_d / var_ref_d
        + (mean_ref_d - mean_recent_d)^2 / var_ref_d
        - 1 + log(var_ref_d / var_recent_d)
    ]
"recent" is P, "reference" is Q -- this direction answers "how much has
what we're seeing now diverged from the lagged baseline."

SELF-CALIBRATION: raw KL has no stable scale (it varies with dataset,
window size, and dimensionality), so it is never thresholded directly.
Instead this class keeps its own running mean/std of the KL values it has
produced since the last reset (plain scalar Welford), and thresholds the
Z-SCORE of each new KL value against that running mean/std. This running
KL-mean/std is NOT updated on a step whose z-score is already at or above
`tau` -- a sustained genuine shift must not be allowed to drag the
"normal" baseline upward and mask itself.

FIRE CONDITION: the z-score must be at or above `tau` for `consecutive`
steps IN A ROW (resets to 0 the moment a step falls back below `tau`), and
only once `warmup_steps` batches have been processed since the last reset.
`warmup_steps` doubles as the refractory period: since reset() returns the
detector to exactly its post-__init__ state, the same warmup gate that
applies at the very start of a run applies identically right after every
later reset, with no separate re-arm mechanism needed.

A fire does exactly one thing from this class's own perspective: it
reports True from step() and records the reason. Everything it actually
MEANS (freeze the live adapter as pending, spawn a fresh one, and reset
this detector) is the caller's responsibility, not this class's -- same
separation of concerns as utils/online_lora_plateau.py's own
LossPlateauDetector, which reports a plateau and lets its caller fold.

max_chunk_samples: an unconditional safety cap with no reference analog,
checked before anything else every step, same placement reasoning and
purpose as LossPlateauDetector's own cap (catches a pathological
never-fires case rather than ever being expected to matter in practice).

ALL NUMERIC DEFAULTS ARE AS SPECIFIED: window ~256 samples / at least 4
batches, warmup = 2 * window_batches (also the refractory period), tau=4
(z-score), consecutive=3 steps, var_eps=1e-4 (variance floor). window_
batches itself is NOT computed by this class -- it takes batches, not
samples, and the caller (which knows its own batch_size) is responsible
for converting a configured sample-count window into a batch count and
making sure it is at least 4. The caller is also responsible for checking
the resulting minimum detectable segment (about 3*window_batches steps)
against the shortest real task in the run, in steps, and shrinking the
window if a task is shorter than that.
"""
import math
from collections import deque

import torch


class KLDriftDetector:
    def __init__(self, window_batches, warmup_multiplier=2, tau=4.0, consecutive=3,
                 var_eps=1e-4, max_chunk_samples=None):
        assert window_batches >= 4, \
            "window_batches must be at least 4 (per spec) -- got {}".format(window_batches)
        self.window_batches = window_batches
        self.warmup_steps = warmup_multiplier * window_batches
        self.tau = tau
        self.consecutive = consecutive
        self.var_eps = var_eps
        self.max_chunk_samples = max_chunk_samples

        self._recent_buf = deque(maxlen=window_batches)
        self._ref_n = None
        self._ref_mean = None
        self._ref_M2 = None
        self._kl_calib_n = None
        self._kl_calib_mean = None
        self._kl_calib_M2 = None
        self._consec = 0
        self._step_count = 0
        self.images_since_fire = 0
        self.last_kl = None
        self.last_z = None
        self.last_fire_reason = None

    def reset(self):
        """Called by the caller right after spawning a new adapter -- NOT by
        this class itself. Returns every piece of state to exactly its
        __init__ value, so the same warmup gate that held at the very start
        of the run holds identically here, doubling as the refractory
        period with no separate mechanism needed."""
        self._recent_buf.clear()
        self._ref_n = None
        self._ref_mean = None
        self._ref_M2 = None
        self._kl_calib_n = None
        self._kl_calib_mean = None
        self._kl_calib_M2 = None
        self._consec = 0
        self._step_count = 0
        self.images_since_fire = 0
        self.last_kl = None
        self.last_z = None

    def step(self, features, batch_size):
        """features: [B, D] tensor of pooled frozen-backbone CLS features
        for this batch (already detached/no_grad -- the caller's
        responsibility). Returns True exactly on the step that should
        spawn a new adapter; the caller does the spawn and then calls
        reset()."""
        self.images_since_fire += batch_size
        if self.max_chunk_samples is not None and self.images_since_fire >= self.max_chunk_samples:
            self._fire("cap")
            return True

        self._step_count += 1
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0, unbiased=False)
        n = features.shape[0]

        # The entry about to be evicted (if the ring buffer is already full)
        # feeds the reference BEFORE it is dropped -- this is what makes the
        # reference lag the window instead of overlapping it.
        if len(self._recent_buf) == self.window_batches:
            old_mean, old_var, old_n = self._recent_buf[0]
            self._update_reference(old_mean, old_var, old_n)
        self._recent_buf.append((batch_mean, batch_var, n))

        self.last_kl = None
        self.last_z = None
        if self._ref_n is None:
            return False

        recent_mean, recent_var, _ = self._pooled_recent_stats()
        ref_var = self._ref_M2 / self._ref_n
        kl = self._kl_diag_gaussian(recent_mean, recent_var, self._ref_mean, ref_var, self.var_eps)
        self.last_kl = kl

        z = None
        if self._kl_calib_n is not None and self._kl_calib_n >= 2:
            std = math.sqrt(max(self._kl_calib_M2 / self._kl_calib_n, 1e-12))
            z = (kl - self._kl_calib_mean) / max(std, 1e-6)
        self.last_z = z

        above = (z is not None) and (z >= self.tau)
        if above:
            self._consec += 1
        else:
            self._consec = 0
            self._update_kl_calib(kl)   # bootstrap (z is None) or normal (z < tau)

        warm = self._step_count >= self.warmup_steps
        if warm and above and self._consec >= self.consecutive:
            self._fire("kl")
            return True
        return False

    def _update_reference(self, mean, var, n):
        """Parallel Welford merge: folds one whole batch's own (mean, var, n)
        into the running reference, treating it as one group rather than n
        individual single-sample updates."""
        m2 = var * n
        if self._ref_n is None:
            self._ref_n, self._ref_mean, self._ref_M2 = n, mean.clone(), m2.clone()
            return
        n_a, mean_a, M2_a = self._ref_n, self._ref_mean, self._ref_M2
        n_b, mean_b, M2_b = n, mean, m2
        n_ab = n_a + n_b
        delta = mean_b - mean_a
        mean_ab = mean_a + delta * (n_b / n_ab)
        M2_ab = M2_a + M2_b + (delta ** 2) * (n_a * n_b / n_ab)
        self._ref_n, self._ref_mean, self._ref_M2 = n_ab, mean_ab, M2_ab

    def _pooled_recent_stats(self):
        total_n = sum(n for _, _, n in self._recent_buf)
        pooled_mean = sum(mean * n for mean, _, n in self._recent_buf) / total_n
        pooled_var = sum(n * (var + (mean - pooled_mean) ** 2)
                          for mean, var, n in self._recent_buf) / total_n
        return pooled_mean, pooled_var, total_n

    def _update_kl_calib(self, kl):
        """Scalar (single-value) Welford update -- each step produces one
        KL number, not a group of samples, unlike _update_reference above."""
        if self._kl_calib_n is None:
            self._kl_calib_n, self._kl_calib_mean, self._kl_calib_M2 = 1, kl, 0.0
            return
        self._kl_calib_n += 1
        delta = kl - self._kl_calib_mean
        self._kl_calib_mean += delta / self._kl_calib_n
        delta2 = kl - self._kl_calib_mean
        self._kl_calib_M2 += delta * delta2

    @staticmethod
    def _kl_diag_gaussian(mean_p, var_p, mean_q, var_q, eps):
        """KL(P || Q) for diagonal Gaussians P=(mean_p,var_p) [recent],
        Q=(mean_q,var_q) [reference], summed over dimensions."""
        var_p = var_p.clamp_min(eps)
        var_q = var_q.clamp_min(eps)
        kl_per_dim = 0.5 * (var_p / var_q + (mean_q - mean_p) ** 2 / var_q - 1.0
                            + torch.log(var_q / var_p))
        return float(kl_per_dim.sum().item())

    def _fire(self, reason):
        self.last_fire_reason = reason
