"""Classifier-alignment (CA) bolt-on, SLCA-style and exemplar-free
(impl_plan_7.27.2026 sec 1.3, extended impl_plan_7.28.2026 sec 2 "CA repair
sweep"). Method-agnostic in principle; wired into SketchLoRA first per the
plan.

v1 (impl_plan_7.27.2026): ClassStats with per-class diagonal covariance,
head-only realignment on pseudo-features. Found to recover +4.6 top1 but cost
-3.7/-5.3 top5 (local 100MB 15T Omni ablation) -- the v2 sweep below exists to
find a variant that keeps the top1 gain without the top5 cost.

v2 variants (impl_plan_7.28.2026 sec 2), all controlled by ClassStats(cov_mode=...)
and align_head(...)'s extra kwargs:
  a. ca_steps sweep {50,100,300} + early stopping on a HELD-OUT pseudo-feature
     batch (drawn once, before training, from the stats snapshot at call time --
     never used for a training step).
  b. covariance: "diag" (v1, per-class diagonal) / "shared_full" (one pooled
     768x768 covariance shared across all classes, ~2.25MB, LDA-style) /
     "low_rank_diag" (per-class rank-8 + diagonal, fit on demand from a bounded
     per-class reservoir of raw features -- NOT a true streaming low-rank
     estimator, a documented simplification: storing O(reservoir_size x 768)
     raw features per class and SVD-ing them at read time is far cheaper to
     implement correctly than an online low-rank covariance update, and stays
     within the same "a few KB/class" memory spirit as the ablation is probing).
  c. real-feature mixing: align_head batches = (1 - real_mix_frac) pseudo +
     real_mix_frac real features drawn from a bounded per-cycle reservoir of
     ACTUAL current-cycle features (collected during training, see
     models/sketchlora.py's _bounded_train_epoch).
  d. logit-adjustment-only: NO head retraining at all. Additive per-class
     PRIOR correction from stored class counts only (Menon et al. 2021 style:
     bias_c = tau * log(pi_c), pi_c = count_c / total) -- user-resolved
     2026-07-28: the plan's "counts/means" phrasing was ambiguous for an
     additive, per-class-CONSTANT correction (a per-sample use of the mean
     would violate the plan's own "cannot damage within-class ranking"
     constraint), resolved to counts-only, the standard well-defined version.
     Applied via apply_logit_adjustment (adds directly to fc.bias, in-place,
     delta-tracked so repeated calls don't compound).
  f. control: mode="diag", real_mix_frac=0, ca_steps as configured -- byte-
     identical to v1's align_head when called with no v2 kwargs.

No SDC-style drift compensation (transporting old means through
sketch_old->sketch_new at fold time) -- documented follow-up, not this round.
"""
import logging

import torch
from torch import nn
from torch.nn import functional as F


class ClassStats:
    def __init__(self, feat_dim, device, cov_mode="diag", reservoir_size=64):
        assert cov_mode in ("diag", "shared_full", "low_rank_diag")
        self.feat_dim = feat_dim
        self.device = device
        self.cov_mode = cov_mode
        self.reservoir_size = reservoir_size
        self.count = {}     # class_id -> int
        self.mean = {}      # class_id -> [feat_dim] running mean
        self.m2 = {}        # class_id -> [feat_dim] running sum-of-squared-deviations (Welford)
        # "shared_full" only: ONE pooled full covariance accumulator, updated
        # from every class's own (evolving) within-class deviation -- the
        # streaming analogue of LDA's pooled covariance estimator.
        self._pooled_count = 0
        self._pooled_m2_full = torch.zeros(feat_dim, feat_dim, device=device)
        # "low_rank_diag" only: bounded per-class reservoir of raw features
        # (reservoir sampling -- uniform over all-seen-so-far, not just recent).
        self._reservoir = {}       # class_id -> [<=reservoir_size, feat_dim]
        self._reservoir_seen = {}  # class_id -> total samples seen (for reservoir sampling)

    @torch.no_grad()
    def update(self, features, labels):
        """features: [B, feat_dim] (detached, penultimate). labels: [B] class ids.
        Welford's online update, per class, per sample in the batch -- exact
        (matches the batch-free sequential-update formula), no retroactive
        recomputation of anything already folded into count/mean/m2."""
        features = features.detach().float()
        labels = labels.detach()
        for c in torch.unique(labels).tolist():
            mask = labels == c
            feats_c = features[mask]
            if c not in self.count:
                self.count[c] = 0
                self.mean[c] = torch.zeros(self.feat_dim, device=self.device)
                self.m2[c] = torch.zeros(self.feat_dim, device=self.device)
                if self.cov_mode == "low_rank_diag":
                    self._reservoir[c] = torch.zeros(0, self.feat_dim, device=self.device)
                    self._reservoir_seen[c] = 0
            for x in feats_c:
                self.count[c] += 1
                delta = x - self.mean[c]
                self.mean[c] = self.mean[c] + delta / self.count[c]
                delta2 = x - self.mean[c]
                self.m2[c] = self.m2[c] + delta * delta2
                if self.cov_mode == "shared_full":
                    self._pooled_count += 1
                    d2 = delta2.unsqueeze(1) @ delta.unsqueeze(0)   # outer product, within-class deviation
                    self._pooled_m2_full = self._pooled_m2_full + d2
                if self.cov_mode == "low_rank_diag":
                    self._reservoir_seen[c] += 1
                    n_seen = self._reservoir_seen[c]
                    res = self._reservoir[c]
                    if res.shape[0] < self.reservoir_size:
                        self._reservoir[c] = torch.cat([res, x.unsqueeze(0)], dim=0)
                    else:
                        # classic reservoir sampling: replace a uniformly-random
                        # slot with probability reservoir_size/n_seen
                        j = torch.randint(0, n_seen, (1,)).item()
                        if j < self.reservoir_size:
                            res[j] = x

    def variance(self, c):
        """Per-class diagonal variance (used directly by cov_mode="diag", and
        as the RESIDUAL diagonal -- after removing the low-rank subspace's own
        captured variance -- by cov_mode="low_rank_diag")."""
        n = self.count[c]
        if n < 2:
            return torch.zeros(self.feat_dim, device=self.device)
        return self.m2[c] / n

    def pooled_covariance(self):
        """cov_mode="shared_full" only: the single pooled 768x768 covariance,
        shared across every class (LDA-style within-class pooled estimator)."""
        if self._pooled_count < 2:
            return torch.eye(self.feat_dim, device=self.device) * 1e-6
        return self._pooled_m2_full / self._pooled_count

    def low_rank_factors(self, c, rank=8):
        """cov_mode="low_rank_diag" only: fit a rank-`rank` + residual-diagonal
        approximation on demand from class c's bounded reservoir. Returns
        (U [feat_dim, r_eff], sigma [r_eff], residual_var [feat_dim]) where
        r_eff <= rank (limited by how many reservoir samples exist so far).
        Falls back to (empty U, empty sigma, the ordinary diag variance) if
        the reservoir has too few samples to fit anything (matches "diag"
        mode's own small-n guard)."""
        res = self._reservoir.get(c)
        full_var = self.variance(c)
        if res is None or res.shape[0] < 2:
            return (torch.zeros(self.feat_dim, 0, device=self.device),
                    torch.zeros(0, device=self.device), full_var)
        centered = res - self.mean[c].unsqueeze(0)
        r_eff = min(rank, centered.shape[0] - 1, self.feat_dim)
        U, S, _ = torch.linalg.svd(centered.t(), full_matrices=False)
        U = U[:, :r_eff]
        # per-direction variance along U, estimated from the reservoir (an
        # (n-1)-sample-biased estimate, standard for a small on-demand fit).
        n = centered.shape[0]
        sigma2 = (S[:r_eff].pow(2) / max(n - 1, 1))
        captured = (U.pow(2) * sigma2.unsqueeze(0)).sum(dim=1)   # variance the low-rank part explains, per input dim
        residual_var = torch.clamp(full_var - captured, min=0.0)
        return U, sigma2, residual_var

    def seen_classes(self):
        return sorted(self.count.keys())

    def memory_bytes(self):
        """M_train ledger reporting, per cov_mode (not enforced here):
          diag:          2 * feat_dim * 4 bytes/class (mean + diagvar)
          shared_full:    (feat_dim * 4 bytes/class for means) + feat_dim^2*4 ONCE (pooled cov)
          low_rank_diag: (feat_dim * 4 bytes/class for means) + reservoir_size*feat_dim*4/class
        """
        n_classes = len(self.count)
        mean_bytes = n_classes * self.feat_dim * 4
        if self.cov_mode == "diag":
            return mean_bytes + n_classes * self.feat_dim * 4
        if self.cov_mode == "shared_full":
            return mean_bytes + self.feat_dim * self.feat_dim * 4
        return mean_bytes + n_classes * self.reservoir_size * self.feat_dim * 4


@torch.no_grad()
def build_low_rank_factor_cache(stats, seen_classes, low_rank=8):
    """Precompute low_rank_factors for every seen class ONCE (an SVD over each
    class's reservoir). stats are frozen during align_head (no .update() calls
    happen inside it), so every one of align_head's ca_steps calls to
    sample_pseudo_features would otherwise recompute the SAME SVD per class
    from scratch -- found via smoke test to make low_rank_diag unusably slow
    (recomputing per INDIVIDUAL SAMPLE draw was ~40x worse still, fixed first,
    but even per-call-not-per-sample was still ca_steps x n_classes redundant
    SVDs of data that never changes across the call). Call once before the
    training loop and pass the result to sample_pseudo_features."""
    return {c: stats.low_rank_factors(c, rank=low_rank) for c in seen_classes}


@torch.no_grad()
def sample_pseudo_features(stats, seen_classes, batch_size, device, low_rank=8,
                            factor_cache=None):
    """Uniformly draw `batch_size` (class, pseudo-feature) pairs, class drawn
    uniformly over seen_classes, feature drawn per stats.cov_mode:
      diag:          N(mu_c, diag(var_c))                          [v1, unchanged]
      shared_full:   N(mu_c, Sigma_pooled)                          [v2 b]
      low_rank_diag: mu_c + U_c z sqrt(sigma2_c) + N(0, diag(resid_var_c))  [v2 b]

    factor_cache: for low_rank_diag, an optional {class: (U, sigma2,
      residual_var)} precomputed by build_low_rank_factor_cache -- avoids
      redoing an SVD per call when stats aren't changing (e.g. across
      align_head's whole training loop). Computed on the fly per class,
      uncached, if not supplied (still correct, just slower).
    """
    idx = torch.randint(0, len(seen_classes), (batch_size,))
    classes = torch.tensor([seen_classes[i] for i in idx.tolist()], device=device)
    feats = torch.empty(batch_size, stats.feat_dim, device=device)

    if stats.cov_mode == "shared_full":
        cov = stats.pooled_covariance()
        # jitter SCALED to the covariance's own magnitude, not a fixed absolute
        # 1e-5 -- found via a real crash (2026-07-28): a fixed-magnitude jitter
        # is negligible regularization when the pooled covariance's own scale
        # is much larger, and can still leave the Cholesky factor badly scaled
        # in near-degenerate directions, producing huge-magnitude pseudo-
        # features -> huge logits -> NaN cross-entropy -> NaN gradients into
        # fc -> (next cycle) NaN propagating through the SHARED forward pass
        # into the adapter's own gradients, corrupting delta_W itself (this is
        # NOT an isolated fc-only failure -- CA's optimizer only ever touches
        # fc.parameters(), but ordinary training's loss backprops through fc
        # INTO the backbone/adapter, so a NaN fc silently poisons the adapter
        # the very next cycle -- confirmed by reproducing the exact SVD
        # non-convergence crash down to "[CA] final_loss=nan" one cycle prior).
        scale = cov.diagonal().mean().clamp(min=1e-6)
        try:
            L = torch.linalg.cholesky(cov + torch.eye(stats.feat_dim, device=device) * scale * 1e-3)
        except RuntimeError:
            L = torch.eye(stats.feat_dim, device=device) * scale.sqrt()
        for i, c in enumerate(classes.tolist()):
            z = torch.randn(stats.feat_dim, device=device)
            feats[i] = stats.mean[c] + L @ z
        return feats, classes

    local_cache = {} if factor_cache is None else factor_cache
    for i, c in enumerate(classes.tolist()):
        mu = stats.mean[c]
        if stats.cov_mode == "low_rank_diag":
            if c not in local_cache:
                local_cache[c] = stats.low_rank_factors(c, rank=low_rank)
            U, sigma2, residual_var = local_cache[c]
            r_eff = U.shape[1]
            z_low = torch.randn(r_eff, device=device) if r_eff > 0 else torch.zeros(0, device=device)
            low_part = (U * sigma2.sqrt().unsqueeze(0)) @ z_low if r_eff > 0 else torch.zeros(stats.feat_dim, device=device)
            std_resid = residual_var.clamp(min=0).sqrt()
            feats[i] = mu + low_part + std_resid * torch.randn(stats.feat_dim, device=device)
        else:   # "diag"
            var = stats.variance(c)
            std = var.clamp(min=0).sqrt()
            feats[i] = mu + std * torch.randn(stats.feat_dim, device=device)
    return feats, classes


@torch.no_grad()
def _sample_held_out_val_batch(stats, seen, batch_size, device, low_rank=8, factor_cache=None):
    """A fixed validation batch drawn ONCE (before any training step) from the
    stats snapshot at call time -- never fed into a training step, used only
    to monitor early-stop loss (impl_plan_7.28.2026 sec 2a)."""
    return sample_pseudo_features(stats, seen, batch_size, device, low_rank=low_rank,
                                   factor_cache=factor_cache)


def align_head(fc, stats, ca_steps, ca_batch, ca_lr, device,
                real_feature_buffer=None, real_mix_frac=0.0,
                early_stop_patience=None, val_batch_size=None, low_rank=8):
    """Head-only realignment over pseudo-features (+ optionally real-feature
    mixing, variant c). Adapters are never touched (this function only ever
    receives/optimizes `fc`'s parameters). Returns a dict of {steps,
    final_loss, stopped_early, val_loss_trace} for the CE-metric ledger and
    the CA-sweep selection metric.

    real_feature_buffer: optional (features [N,feat_dim], labels [N]) tensor
      pair of ACTUAL current-cycle features (collected during training) --
      variant c. Ignored if real_mix_frac <= 0.
    early_stop_patience: if set (variant a), stop once val loss (measured on a
      FIXED held-out pseudo-feature batch, drawn once before training) hasn't
      improved for this many consecutive checks (checked every 10 steps).
    """
    seen = stats.seen_classes()
    if len(seen) < 1:
        return {"steps": 0, "final_loss": None, "stopped_early": False, "val_loss_trace": []}
    optimizer = torch.optim.AdamW(fc.parameters(), lr=ca_lr)

    # stats are frozen for this whole call (no .update() happens inside
    # align_head) -- precompute low_rank_diag's per-class SVD ONCE rather than
    # once per one of ca_steps calls to sample_pseudo_features (found via
    # smoke test: recomputing it every call made this mode unusably slow).
    factor_cache = build_low_rank_factor_cache(stats, seen, low_rank) \
        if stats.cov_mode == "low_rank_diag" else None

    val_feats = val_classes = None
    if early_stop_patience is not None:
        vbs = val_batch_size or ca_batch
        val_feats, val_classes = _sample_held_out_val_batch(
            stats, seen, vbs, device, low_rank=low_rank, factor_cache=factor_cache)

    def _masked_loss(logits, classes):
        mask = torch.full_like(logits, float("-inf"))
        mask[:, seen] = 0.0
        return F.cross_entropy(logits + mask, classes)

    final_loss = None
    val_loss_trace = []
    best_val, since_best, stopped_early = float("inf"), 0, False
    n_real = int(round(ca_batch * real_mix_frac)) if real_feature_buffer is not None else 0
    n_real = min(n_real, real_feature_buffer[0].shape[0]) if real_feature_buffer is not None else 0
    n_pseudo = ca_batch - n_real

    for step in range(ca_steps):
        feats_p, classes_p = sample_pseudo_features(stats, seen, max(n_pseudo, 1), device,
                                                       low_rank=low_rank, factor_cache=factor_cache)
        if n_real > 0:
            real_feats, real_labels = real_feature_buffer
            ridx = torch.randint(0, real_feats.shape[0], (n_real,))
            feats = torch.cat([feats_p[:n_pseudo], real_feats[ridx].to(device)], dim=0)
            classes = torch.cat([classes_p[:n_pseudo], real_labels[ridx].to(device)], dim=0)
        else:
            feats, classes = feats_p, classes_p
        logits = fc(feats)["logits"]   # SimpleLinear.forward returns {"logits": ...}, not a raw tensor
        loss = _masked_loss(logits, classes)
        loss_val = loss.item()
        if not torch.isfinite(loss).item():
            # Defensive backstop (2026-07-28, found via a real crash): a NaN/Inf
            # step here (e.g. shared_full sampling a pathological pseudo-feature
            # batch) must NEVER reach optimizer.step() -- fc has no other
            # protection, and a single NaN update permanently poisons fc for
            # the rest of the run. Worse, ordinary training's loss backprops
            # THROUGH fc into the backbone/adapter every subsequent cycle, so a
            # poisoned fc silently corrupts the adapter's own gradients next
            # cycle too (confirmed: this is what caused a "SVD failed to
            # converge" crash in _compress(), one cycle after an unguarded
            # NaN CA step -- not a rand_svd robustness gap by itself). Skip
            # this step entirely, keep fc's last good weights, and continue.
            logging.warning("[CA] non-finite loss (%s) at step %d -- skipping "
                             "this step's optimizer update", loss_val, step)
            final_loss = loss_val
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss_val

        if early_stop_patience is not None and (step + 1) % 10 == 0:
            with torch.no_grad():
                val_logits = fc(val_feats)["logits"]
                val_loss = _masked_loss(val_logits, val_classes).item()
            val_loss_trace.append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val, since_best = val_loss, 0
            else:
                since_best += 1
            if since_best >= early_stop_patience:
                stopped_early = True
                break

    return {"steps": step + 1, "final_loss": final_loss,
            "stopped_early": stopped_early, "val_loss_trace": val_loss_trace}


# -- variant d: logit-adjustment-only (no head retraining at all) -----------
def logit_adjust_bias(stats, tau=1.0):
    """Menon et al. 2021-style additive per-class prior correction:
    bias_c = tau * log(pi_c), pi_c = count_c / total_count. Counts-only
    (2026-07-28 user resolution of the plan's ambiguous "counts/means"
    phrasing) -- a per-class CONSTANT, so it shifts each class's decision
    threshold uniformly without touching within-class feature ranking."""
    seen = stats.seen_classes()
    total = sum(stats.count[c] for c in seen)
    bias = torch.zeros(max(seen) + 1, device=stats.device)
    for c in seen:
        pi_c = stats.count[c] / total
        bias[c] = tau * torch.log(torch.tensor(pi_c, device=stats.device))
    return bias


def apply_logit_adjustment(fc, stats, tau, prev_correction):
    """Applies the DELTA between this cycle's target bias correction and the
    previously-applied one directly to fc.bias (in-place, additive) -- no
    optimizer, no gradient step, "no head retraining" per the plan. Delta-
    tracked (not a fresh add every cycle) so repeated calls across cycles
    don't compound past the current target. Returns the new correction
    vector to pass as `prev_correction` next time (starts at None/zeros)."""
    target = logit_adjust_bias(stats, tau)
    n = target.shape[0]
    if prev_correction is None or prev_correction.shape[0] < n:
        prev = torch.zeros(n, device=stats.device)
        if prev_correction is not None:
            prev[:prev_correction.shape[0]] = prev_correction
    else:
        prev = prev_correction
    delta = target - prev
    with torch.no_grad():
        fc.bias.data[:n] += delta.to(fc.bias.dtype)
    return target
