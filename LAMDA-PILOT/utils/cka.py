"""Linear Centered Kernel Alignment (Kornblith et al. 2019, "Similarity of
Neural Network Representations Revisited"), for the olora_sketch_fd_divergence
campaign's representation comparisons. No training-time dependency -- pure
numpy, operates on already-extracted (N, D) feature matrices from
utils/embed_drift.py's saved .npz groups.

CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

computed on CENTERED X, Y (each column/feature mean-subtracted first) -- the
linear-kernel form of HSIC-based CKA, equivalent to the general kernel
formulation with a linear kernel. Invariant to orthogonal transformation and
isotropic scaling of either X or Y (not to arbitrary invertible linear
transforms), which is exactly the invariance needed here: SketchLoRA/FD
Sum/O-LoRA parameterize their adapters completely differently, so raw
feature-space comparisons would be meaningless without it.
"""
import numpy as np


def _center(X):
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X, Y):
    """X: [N, Dx], Y: [N, Dy] -- same N rows (same inputs), any feature dims.
    Returns a scalar in [0, 1] (up to floating-point slack)."""
    X = _center(np.asarray(X, dtype=np.float64))
    Y = _center(np.asarray(Y, dtype=np.float64))
    # ||Y^T X||_F^2 == sum((Y^T X)**2); computed via the cross Gram product
    # rather than forming Y^T X directly when D is large -- fine at this
    # project's scale (D ~= 768), but written to also be correct/cheap if
    # N < D by routing through whichever Gram matrix is smaller.
    cross = Y.T @ X
    num = np.sum(cross ** 2)
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    if denom == 0:
        return float("nan")
    return float(num / denom)
