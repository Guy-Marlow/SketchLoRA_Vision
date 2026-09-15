"""Decoupled-space Frequent-Directions merge for SketchLoRA (2026-09-15 user
design). merge_op="freqdir" in models/sketchlora.py.

Standard SketchLoRA (merge_op="randsvd"/"exactsvd") forms the dense update
delta_W = B_hat @ A_hat (sketch + residual, summed) and decomposes THAT
directly -- so the new sketch's input directions (A_hat, from the right
singular vectors) and output directions (B_hat, from the left singular
vectors) are extracted from the SAME object and are therefore coupled: they
"know about" each other by construction.

freqdir tests what happens when that coupling is severed. The sketch's and
residual's INPUT factors (A) and OUTPUT factors (B) are concatenated and
compressed SEPARATELY -- the dense [d_out, d_in] matrix is never formed, not
even once:

  M_A = [A_sketch ; A_residual]   (2r, d_in)   -- concat along the rank axis
  M_B = [B_sketch | B_residual]   (d_out, 2r)   -- concat along the rank axis

Each is compressed via the classic Frequent-Directions shrink (Liberty 2013,
"Simple and Deterministic Matrix Sketching"; see also Ghashami et al. SICOMP
2016 for the streaming-sketch error bound this generalizes): square the
singular values, subtract the SMALLEST squared value from every one of them
(zeroing at least the last entry, more on ties -- this is the "pay rent"
step that gives FD its bounded-error guarantee), take the square root. This
uniform subtraction preserves the DESCENDING ORDER of the spectrum, so it
does not change which top-r_hat directions get kept relative to plain
truncation -- only their magnitude, debiased downward by the smallest
observed "noise floor" energy.

The resulting new A_hat and B_hat come from two entirely independent
decompositions, paired only by rank position (the i-th largest direction of
M_A's compression next to the i-th largest of M_B's) -- there is no
mathematical reason those should correspond beyond both surviving their own
independent top-r_hat cut. Severing that correspondence is the entire point
of the ablation.

NOT to be confused with the pre-existing self.fd_shrinkage bolt-on
(utils/fd.py::apply_fd_shrinkage), which shrinks a SINGLE joint decomposition
AFTER its target rank is already chosen, by the energy of the first
DISCARDED direction at that rank (sigma_{r_hat+1}^2) -- a different
reference point than the one used here (the smallest value in the full,
pre-truncation 2r-length stack, sigma_{2r}^2), applied to a fundamentally
different (joint, not decoupled) decomposition. The two are NOT composed;
freqdir's shrink is intrinsic to its own construction, not an optional
bolt-on -- see the fd_shrinkage-incompatibility warning in
models/sketchlora.py's __init__, which already covers merge_op="freqdir"
via its existing "not in (randsvd, exactsvd)" check.
"""
import torch


def freqdir_shrink(S):
    """Square every singular value, subtract the smallest squared value from
    all of them, take the square root. Returns a full-length (same shape as
    S), still-descending spectrum AT SINGULAR-VALUE SCALE (not squared) --
    NOT yet truncated to a target rank and not yet re-split onto singular
    vectors. The caller truncates to r_hat and splits sqrt(this) onto U or
    Vh, matching utils/randsvd.py::factors_from_probe's own convention
    (root_S = S[:r_hat].sqrt(); B_hat = U*root_S; A_hat = root_S*Vh) -- here
    applied one side at a time, since each freqdir decomposition (of M_A or
    of M_B) only ever feeds ONE of the two factors, never both.

    S must be sorted descending (as torch.linalg.svd already returns it).
    """
    S_sq = S.pow(2)
    rent = S_sq[-1]
    return torch.clamp(S_sq - rent, min=0.0).sqrt()


def freqdir_input_factor(M_A, r_hat):
    """Compress the concatenated input-space matrix M_A = [A_sketch; A_res]
    (shape [2r, d_in]) down to the new sketch's A_hat (shape [r_hat, d_in])
    via freqdir_shrink + truncation, split onto the right singular vectors
    (the "A" half of factors_from_probe's construction)."""
    _, S, Vh = torch.linalg.svd(M_A, full_matrices=False)
    S_shrunk = freqdir_shrink(S)
    assert S_shrunk[r_hat - 1] > 0, (
        "freqdir: target rank r_hat={} reaches into the FD-zeroed tail of the "
        "input-space spectrum (length {}) -- shrinkage should only ever zero a "
        "handful of directions at the very bottom of a {}-length stack; this "
        "would mean r_hat is not comfortably below that, which should not "
        "happen at r_hat=svd_rank, 2r=2*svd_rank.".format(
            r_hat, S.numel(), S.numel()))
    root = S_shrunk[:r_hat].sqrt()
    return root.unsqueeze(1) * Vh[:r_hat, :]


def freqdir_output_factor(M_B, r_hat):
    """Compress the concatenated output-space matrix M_B = [B_sketch | B_res]
    (shape [d_out, 2r]) down to the new sketch's B_hat (shape [d_out, r_hat])
    via freqdir_shrink + truncation, split onto the left singular vectors
    (the "B" half of factors_from_probe's construction)."""
    U, S, _ = torch.linalg.svd(M_B, full_matrices=False)
    S_shrunk = freqdir_shrink(S)
    assert S_shrunk[r_hat - 1] > 0, (
        "freqdir: target rank r_hat={} reaches into the FD-zeroed tail of the "
        "output-space spectrum (length {}) -- see freqdir_input_factor's "
        "assert message for why this should not happen.".format(r_hat, S.numel()))
    root = S_shrunk[:r_hat].sqrt()
    return U[:, :r_hat] * root.unsqueeze(0)
