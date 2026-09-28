"""Frequent-Directions shrinkage applied to the DENSE composite update's
randomized-SVD probe (2026-09-28 user design) -- SketchLoRA merge_op=
"freqdir_dense". Distinct from BOTH:
  * merge_op="freqdir" (utils/freqdir.py): never forms delta_W at all --
    sketches the concatenated up/down-projection matrices INDEPENDENTLY,
    which is exactly what this variant was designed to avoid (the user's
    diagnosis: independent A/B sketches drift apart from each other,
    producing large accuracy losses).
  * fd_shrinkage=True (utils/fd.py's apply_fd_shrinkage): shrinks an
    ALREADY-TRUNCATED kept spectrum by the energy of the first DISCARDED
    singular value, post hoc. This module instead shrinks the FULL,
    UNTRUNCATED randomized-probe candidate set by its OWN smallest member
    BEFORE truncating -- the classic streaming-FD buffer-shrink step
    (Liberty 2013 / Ghashami et al. SICOMP 2016), just applied once per
    merge instead of incrementally.

Used from models/sketchlora.py's merge_op="freqdir_dense" branch: build
delta_W = B_s@A_s + B_r@A_r exactly as merge_op="randsvd" does, run the SAME
rand_svd_probe over it, then call fd_shrink_probe on the resulting candidate
spectrum instead of just slicing the top svd_rank values straight out. Does
NOT truncate -- mirrors rand_svd_probe's own convention (return the full,
untruncated spectrum; the caller picks r_hat and slices), so callers can still
read the shrunk value one past the cutoff for diagnostics (sigma_next).
"""
import torch


def fd_shrink_probe(S):
    """Square every candidate singular value in S, subtract the SMALLEST
    squared value from all of them (so the smallest becomes exactly 0 --
    guaranteed by construction, not approximately), take the square root.
    S must be sorted descending (rand_svd_probe's own convention) and
    untruncated (the full working_rank+oversampling candidate set, not
    already sliced to a target rank).

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
