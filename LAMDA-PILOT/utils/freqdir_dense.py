"""Frequent-Directions shrinkage applied to the DENSE composite update's
randomized-SVD probe, restricted to its non-padding window (2026-09-28 user
design; index-window fix 2026-09-29, after a brief exact-SVD detour) --
SketchLoRA merge_op="freqdir_dense". Distinct from BOTH:
  * merge_op="freqdir" (utils/freqdir.py): never forms delta_W at all --
    sketches the concatenated up/down-projection matrices INDEPENDENTLY,
    which is exactly what this variant was designed to avoid (the user's
    diagnosis: independent A/B sketches drift apart from each other,
    producing large accuracy losses).
  * fd_shrinkage=True (utils/fd.py's apply_fd_shrinkage): shrinks an
    ALREADY-TRUNCATED kept spectrum by the energy of the first DISCARDED
    singular value, post hoc. This module instead shrinks a full,
    UNTRUNCATED candidate set by its OWN smallest member BEFORE truncating
    -- the classic streaming-FD buffer-shrink step (Liberty 2013 / Ghashami
    et al. SICOMP 2016), just applied once per merge instead of
    incrementally.

Used from models/sketchlora.py's merge_op="freqdir_dense" branch: build
delta_W = B_s@A_s + B_r@A_r exactly as merge_op="randsvd" does, run the SAME
rand_svd_probe over it (same working_rank/oversampling), restrict the
returned spectrum to S[:composite_rank] (composite_rank = prev_rank+
residual_total, an EXACT upper bound on delta_W's true rank -- and, because
working_rank = max(composite_rank, svd_rank) >= composite_rank always, this
window is exactly the portion the randomized projection recovers accurately;
everything beyond it is the oversampling padding, probing directions
delta_W has no true component in), then call fd_shrink_probe on that
restricted spectrum instead of just slicing the top svd_rank values straight
out.

ORIGINAL VERSION (no index restriction -- fd_shrink_probe read the ENTIRE
probe, oversampling padding included) VERIFIED BROKEN 2026-09-29: the
padding -- and hence "the smallest candidate value" fd_shrink_probe
subtracted -- was always the randomized-projection noise floor (~1e-16 to
1e-17 squared, confirmed in the actual freqdir_dense_wave1 campaign's own
sketch_diag output, every fold), making the shrinkage step a total no-op
(results were bit-identical to plain randsvd). A brief intermediate fix
switched to an exact (non-randomized) SVD of delta_W with the same
S[:composite_rank] restriction, which also worked, but departed further
than necessary from the original "identical to vanilla SketchLoRA/randsvd"
design -- restricting the RANDOMIZED probe's own index range achieves the
same fix while keeping that original design intact. Once composite_rank >
svd_rank (every fold after the first, in fixed-rank mode), indices
[svd_rank:composite_rank] are real, accurately-recovered singular values,
not noise, so the smallest of them is a genuine "rent" charge.

fd_shrink_probe itself is decomposition-method-agnostic -- it just needs "a
candidate spectrum, not yet truncated to final rank," whatever produced it.
Does NOT truncate -- so callers can still read the shrunk value one past the
cutoff for diagnostics (sigma_next).
"""
import torch


def fd_shrink_probe(S):
    """Square every candidate singular value in S, subtract the SMALLEST
    squared value from all of them (so the smallest becomes exactly 0 --
    guaranteed by construction, not approximately), take the square root.
    S must be sorted descending and untruncated (the full candidate set --
    e.g. a randomized-SVD probe's spectrum restricted to the composite's
    true-rank bound -- not already sliced to a target rank).

    Returns (S_shrunk, stats) -- S_shrunk is the same length as S, still
    descending (subtracting a constant from every squared value preserves
    order); stats records the shrinkage amount and the pre/post energy of
    the full candidate set.
    """
    sq = S.pow(2)
    delta = sq.min()
    shrunk_sq = torch.clamp(sq - delta, min=0.0)
    S_shrunk = shrunk_sq.sqrt()
    stats = {
        "delta": delta.item(),
        "pre_shrink_energy": sq.sum().item(),
        "post_shrink_energy": shrunk_sq.sum().item(),
        "n_candidates": int(S.numel()),
    }
    return S_shrunk, stats
