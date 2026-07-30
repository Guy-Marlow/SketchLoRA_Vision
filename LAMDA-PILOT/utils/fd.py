"""Frequent-Directions-style shrinkage bolt-on for SketchLoRA (impl_plan_7.27.2026
sec 1.1). Applied INSIDE a merge, after the eviction/rank-selection logic has
already chosen the kept rank l and the merge algorithm has produced a factor
pair (B_hat, A_hat) approximating the rank-l truncated SVD of delta_W. Shrinks
every KEPT squared singular value by the energy of the FIRST DISCARDED
singular value (classic FD "pay rent" step), which upper-bounds the growth of
the accumulated sketch instead of letting it grow monotonically -- see Liberty
2013 / Ghashami et al. SICOMP 2016 for the streaming-matrix-sketching result
this generalizes, and Mroueh et al. AISTATS 2017 (Co-occurring Directions) for
the two-sided/product form relevant to a B@A adapter factorization.

Deliberately merge_op-agnostic in its math (only needs the exact singular
values S of delta_W, already computed elsewhere for diagnostics/rank-selection
whenever sketch_diag or an adaptive energy_target is active -- never a second
SVD), but the "kept factor has norm sqrt(sigma_i)" assumption it rescales
against is only exactly true for exactsvd; for randsvd it holds approximately
(the randomized top-rank estimate is close to the true spectrum by
construction). Scoped to merge_op in {"randsvd", "exactsvd"} -- naive_sum and
countsketch are not truncated-SVD algorithms, so "Sigma[l]" has no meaning for
them; caller should skip the call (no-op) rather than pass through those ops.
"""
import torch


def apply_fd_shrinkage(B_hat, A_hat, S, r_hat_t):
    """Shrink the kept singular values by the first-discarded singular value's
    energy, then rescale (B_hat, A_hat) proportionally so their product's
    effective singular values become sigma_shrunk instead of the original
    Sigma[:r_hat_t].

    Args:
      B_hat, A_hat: the already-computed merged factor pair, [d, l] / [l, d].
      S: exact singular values of delta_W (descending), from
         torch.linalg.svdvals(delta_W) -- the same full spectrum already
         computed for rank-selection/diagnostics, never recomputed here.
      r_hat_t: kept rank l (== B_hat.shape[1]).

    Returns:
      (B_shrunk, A_shrunk, stats) where stats has pre_shrink_energy,
      post_shrink_energy, rent (= sigma_{l+1}^2, the discarded energy charged
      per kept direction), and n_kept.
    """
    kept = S[:r_hat_t]                                   # Sigma[:l], descending, >=0
    pre_energy = kept.pow(2).sum()
    if S.numel() > r_hat_t:
        sigma_discarded = S[r_hat_t]                     # Sigma[l], the FIRST discarded value
    else:
        # nothing discarded -- rent is 0, shrinkage is an exact no-op (plan sec 1.1)
        sigma_discarded = torch.zeros((), dtype=S.dtype, device=S.device)
    rent = sigma_discarded.pow(2)
    sigma_shrunk = torch.sqrt(torch.clamp(kept.pow(2) - rent, min=0.0))
    post_energy = sigma_shrunk.pow(2).sum()

    # per-kept-direction rescale ratio: sqrt(sigma_shrunk_i / sigma_i), 0 where
    # sigma_i itself is ~0 (a zero direction stays zero, nothing to shrink).
    eps = torch.finfo(kept.dtype).tiny if kept.dtype.is_floating_point else 1e-30
    safe_kept = torch.clamp(kept, min=eps)
    ratio = torch.sqrt(torch.clamp(sigma_shrunk / safe_kept, min=0.0))
    ratio = torch.where(kept > eps, ratio, torch.zeros_like(ratio))

    B_shrunk = B_hat * ratio.to(B_hat.dtype).unsqueeze(0)   # scale columns of B_hat [d, l]
    A_shrunk = A_hat * ratio.to(A_hat.dtype).unsqueeze(1)   # scale rows of A_hat    [l, d]

    stats = {
        "pre_shrink_energy": pre_energy.item(),
        "post_shrink_energy": post_energy.item(),
        "rent": rent.item(),
        "n_kept": int(r_hat_t),
    }
    return B_shrunk, A_shrunk, stats
