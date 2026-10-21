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

2026-10-14 RE-DERIVATION (ell vs k): briefly "fixed" to read delta from
position svd_rank instead of S.min() (i.e. treating composite_rank as a
Fast-FrequentDirections 2*ell buffer with ell=svd_rank), on the reasoning
that Fast-FD's Algorithm 3.1 drives positions [ell, 2ell) to zero via
delta=sigma_ell^2. That reasoning doesn't hold here: the FD paper's bound is
stated for ANY k < ell --
  ||A-pi_{B_k}(A)||_F^2 <= (1 + k/(ell-k)) ||A-A_k||_F^2
-- where B is the full ell-wide maintained sketch (shrunk by ITS OWN
boundary value sigma_ell^2, i.e. ell = the width of the candidate set being
shrunk) and B_k is a PLAIN top-k truncation of that already-shrunk B, with
NO further shrinking. That is exactly this module's original design:
composite_rank plays ell (the full candidate set fd_shrink_probe receives),
svd_rank plays k (the truncation applied after, in models/sketchlora.py,
with no additional shrinking). S.min() over the full S[:composite_rank]
window IS sigma_ell^2 for that window -- the "fix" instead moved delta to
position k, which only matches the narrower special case ell=k (Fast-FD's
own buffer-doubling recipe for maintaining a sketch whose OUTPUT width is
ell), not the general k<ell bound this design was always relying on.
Reverted; S.min() restored.

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

    S plays "ell" (the full FD-maintained sketch width) in Liberty 2013 /
    Ghashami et al. SICOMP 2016's notation -- this shrinks the WHOLE
    candidate set by sigma_ell^2 = S.min()**2, exactly Algorithm 2.1/3.1's
    own-width shrink. Callers that then truncate the result to some k <
    S.numel() (as models/sketchlora.py's freqdir_dense branch does, to
    svd_rank) get the paper's general k<ell bound
    (||A-pi_{B_k}(A)||_F^2 <= (1+k/(ell-k))||A-A_k||_F^2) on that
    truncation for free -- no further shrinking needed or wanted at the k
    boundary.

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
