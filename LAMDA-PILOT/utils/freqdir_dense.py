"""Frequent-Directions shrinkage applied to the DENSE composite update's
EXACT spectrum (2026-09-28 user design; EXACT-SVD fix 2026-09-29) --
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
delta_W = B_s@A_s + B_r@A_r exactly as merge_op="randsvd" does, decompose it
with torch.linalg.svd (exact, not randomized), restrict to S[:composite_rank]
(composite_rank = prev_rank+residual_total, an EXACT upper bound on delta_W's
true rank -- everything beyond it is structurally ~0, not real content), then
call fd_shrink_probe on that restricted spectrum instead of just slicing the
top svd_rank values straight out.

ORIGINAL VERSION (randomized rand_svd_probe, no restriction) VERIFIED BROKEN
2026-09-29: rand_svd_probe's working_rank was itself sized to exactly bound
delta_W's true rank, so its oversampling padding -- and hence "the smallest
candidate value" fd_shrink_probe subtracted -- was always the randomized-
projection noise floor (~1e-16 to 1e-17 squared, confirmed in the actual
freqdir_dense_wave1 campaign's own sketch_diag output, every fold), making
the shrinkage step a total no-op (results were bit-identical to plain
randsvd). Restricting to the exact spectrum's S[:composite_rank] fixes this:
once composite_rank > svd_rank (every fold after the first, in fixed-rank
mode), indices [svd_rank:composite_rank] are REAL discarded singular values,
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
    e.g. an exact SVD's spectrum restricted to the composite's true-rank
    bound -- not already sliced to a target rank).

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
