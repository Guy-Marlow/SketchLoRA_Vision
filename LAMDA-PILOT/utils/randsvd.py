
import torch
import numpy as np

'''inputs:
    #M: m x n matrix (either a torch.Tensor or an np.ndarray)
    #target_rank: int
    #oversampling: int
'''
def rand_svd(M: np.ndarray | torch.Tensor, target_rank: int, oversampling: int):
    print(f'Target rank: {target_rank}')
    
    if(isinstance(M, np.ndarray)):
        omega = np.random.randn(M.shape[1], target_rank + oversampling)
        Y = M @ omega
        Q = np.linalg.qr(Y).Q
        M_bar = np.transpose(Q) @ M

        U_bar, S, Vh = np.linalg.svd(M_bar)
        S = np.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = np.power(S, 0.5)[0:target_rank, 0:target_rank]

        B_hat = U @ S_root
        A_hat = S_root @ Vh[0:target_rank, :]

        return (B_hat, A_hat)
    else:
        M = M.cuda()

        omega = torch.randn(M.shape[1], target_rank + oversampling, device=M.device)
        Y = M @ omega
        Q, _ = torch.linalg.qr(Y)
        M_bar = Q.t() @ M
        try:
            U_bar, S, Vh = torch.linalg.svd(M_bar)
        except torch._C._LinAlgError:
            # 2026-07-28: the default cusolver driver's heuristic algorithm can
            # fail to converge on M_bar (small, but occasionally ill-conditioned
            # or near-repeated-singular-value) -- reproduced deterministically
            # (same seed, same crash point) during the CA v2 sweep's shared_full
            # covariance run, unrelated to CA itself (CA never touches this
            # code path; the failure is in the ordinary sketch-fold SVD).
            # driver="gesvd" is the classical Golub-Kahan QR algorithm --
            # slower but far more numerically robust, and M_bar is tiny
            # (target_rank+oversampling square-ish), so the slowdown is
            # negligible here. Retried once; if THIS also fails, let it raise.
            U_bar, S, Vh = torch.linalg.svd(M_bar, driver="gesvd")

        S = torch.diag(S)
        U = (Q @ U_bar)[:, 0:target_rank]
        S_root = torch.pow(S, 0.5)[0:target_rank, 0:target_rank]

        B_hat = U @ S_root
        A_hat = S_root @ Vh[0:target_rank, :]

        return (B_hat, A_hat)


# *** UNTESTED as of 2026-08-03 *** -- FIX for a real correctness bug (user-
# flagged, "catastrophic"): models/sketchlora.py's adaptive-rank (energy_target)
# path was calling torch.linalg.svdvals(delta_W) -- an EXACT SVD of the full
# matrix -- to decide the target rank r_hat_t, THEN separately calling rand_svd()
# above (its OWN independent randomized decomposition) to build the actual
# (B_hat, A_hat) factors. That means the rank decision used perfect knowledge of
# the true spectrum that a randomized method is never supposed to have access to
# (a real, accuracy-affecting deviation from the algorithm as intended -- not
# just a wasted-compute issue), AND paid for two unrelated SVDs every fold when
# exactly one was ever supposed to happen.
#
# rand_svd_probe splits rand_svd's own algorithm into two steps: (1) run the
# SAME randomized projection + QR + exact-SVD-of-the-small-reduced-matrix,
# computing enough directions to safely cover the composite's true rank (an
# EXACT upper bound derivable from the LoRA structure itself -- prev_rank +
# residual_total, no estimation needed), returning the FULL, UNTRUNCATED (U, S,
# Vh); (2) the caller (models/sketchlora.py::_compress) picks r_hat_t from
# THIS S (the randomized method's own approximate spectrum, not the exact one),
# then slices U/S/Vh to r_hat_t to build the final factors -- the identical
# construction rand_svd() uses internally, just with the truncation point
# decided AFTER the (one) decomposition instead of required as an input before
# it. Exactly one SVD (on the small M_bar), on the smaller matrix, matching how
# the algorithm is supposed to work.
#
# working_rank MUST be a valid upper bound on M's true rank (see the caller) --
# same requirement rand_svd() already has for target_rank+oversampling, just
# made explicit here since working_rank is now determined by the CALLER's
# knowledge of the matrix's structure rather than being "whatever rank we
# already decided we want."
def rand_svd_probe(M: torch.Tensor, working_rank: int, oversampling: int):
    M = M.cuda()
    omega = torch.randn(M.shape[1], working_rank + oversampling, device=M.device)
    Y = M @ omega
    Q, _ = torch.linalg.qr(Y)
    M_bar = Q.t() @ M
    try:
        U_bar, S, Vh = torch.linalg.svd(M_bar)
    except torch._C._LinAlgError:
        # same cusolver-heuristic-non-convergence fallback as rand_svd() above.
        U_bar, S, Vh = torch.linalg.svd(M_bar, driver="gesvd")
    U = Q @ U_bar   # NOT truncated -- caller picks r_hat_t and slices
    return U, S, Vh


def factors_from_probe(U, S, Vh, r_hat):
    """Truncate a rand_svd_probe() decomposition to r_hat directions, building
    (B_hat, A_hat) with the SAME construction rand_svd() uses internally
    (S_root = sqrt(S[:r_hat]); B_hat = U[:,:r_hat] * S_root; A_hat = S_root *
    Vh[:r_hat,:]) -- kept as a small helper so this exact formula is written
    once, not duplicated at every call site that needs to slice a probe."""
    root_S = S[:r_hat].sqrt()
    B_hat = U[:, :r_hat] * root_S.unsqueeze(0)
    A_hat = root_S.unsqueeze(1) * Vh[:r_hat, :]
    return B_hat, A_hat


def random_factors_from_probe(U, S, Vh, r_hat, composite_rank, seed):
    """Ablation counterpart to factors_from_probe (2026-09-02, "random rank
    selection" sensitivity study): instead of keeping the TOP r_hat directions
    by singular-value magnitude (the standard, principled choice), keeps a
    RANDOM r_hat-sized subset of the first `composite_rank` directions.
    composite_rank is delta_W's true, meaningful rank (prev_rank +
    residual_total in models/sketchlora.py) -- columns beyond that, out to
    rand_svd_probe's working_rank+oversampling, are numerical padding from the
    randomized projection, not real content, and MUST be excluded from the
    random pool (matches the same exclusion the "retain" admission rule
    already applies to this same S for the identical reason -- see
    models/sketchlora.py's _compress).

    Seeded with a torch.Generator local to this call, never the global RNG --
    this ablation must not perturb any other randomness (weight init,
    dataloader shuffling, rand_svd's own projection, ...) elsewhere in the
    same run. The caller is responsible for deriving a seed that varies per
    (task, layer, projection) the same way countsketch_compress's own seed
    does, so consecutive merges/modules don't all draw the identical subset.

    Exists to let the project empirically test whether keeping the LARGEST
    singular values specifically -- rather than an arbitrary same-size subset
    of the meaningful spectrum -- is actually what makes truncated-SVD
    compression good, rather than just "keeping some fixed number of
    directions" being sufficient on its own."""
    assert r_hat <= composite_rank, \
        "cannot keep {} random directions out of only {} meaningful ones".format(
            r_hat, composite_rank)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(composite_rank, generator=gen)
    keep = perm[:r_hat].to(U.device)
    root_S = S[keep].sqrt()
    B_hat = U[:, keep] * root_S.unsqueeze(0)
    A_hat = root_S.unsqueeze(1) * Vh[keep, :]
    return B_hat, A_hat


def rand_svd_debug(M: np.ndarray, target_rank: int, oversampling: int):

    omega = np.random.randn(M.shape[1], target_rank + oversampling)
    # print(np.round(omega, 2))
    # print(omega.shape)
    # print("\n")


    #y: m x (target_rank + oversampling)
    # print("RANDOM PROJECTION TO REDUCED RANK")

    Y = M @ omega
    # print(np.round(Y, 2))
    # print(Y.shape)
    # print("\n")


    #QR decomp of Y; retain only orthonormal basis Q
    # print("QR DECOMP OF Y")
    Q = np.linalg.qr(Y).Q
    # print(np.round(Q, 2))
    # print(Q.shape)
    # print("\n")


    #dimension-reduce M: (target_rank + oversampling) x n
    # print("DIMENSION REDUCTION OF M")
    M_bar = np.transpose(Q) @ M
    # print(np.round(M_bar, 2))
    # print(M_bar.shape)
    # print("\n")


    #SVD on small matrix M_bar
    # print("SVD ON REDUCED MATRIX")
    U_bar, S, Vh = np.linalg.svd(M_bar)
    S = np.diag(S)

    # print(np.round(U_bar, 2))
    # print(f'{U_bar.shape}\n')
    # print(np.round(S, 2))
    # print(f'{S.shape}\n')
    # print(np.round(Vh, 2))
    # print(Vh.shape)
    # print("\n")


    #Left singular vectors in original space
    # print("RETURN LEFT SINGULAR VECTORS TO ORIGINAL SPACE")
    U = (Q @ U_bar)[:, 0:target_rank]
    # print(np.round(U, 2))
    # print(U.shape)
    # print("\n")


    #Express LoRA factors; B_hat @ A_hat ~= M
    # print("PREPARE SINGULAR VECTOR")
    S_root = np.power(S, 0.5)[0:target_rank, 0:target_rank]
    # print(np.round(S_root, 2))
    # print(S_root.shape)
    # print("\n")


    # print("LORA FACTOR B:")
    B_hat = U @ S_root
    # print(np.round(B_hat, 2))
    # print(B_hat.shape)
    # print("\n")


    # print("LORA FACTOR A:")
    # print(S_root.shape)
    # print(Vh.shape)
    # print(Vh[:, 0:target_rank].shape)
    A_hat = S_root @ Vh[0:target_rank, :]
    # print(np.round(A_hat, 2))
    # print(A_hat.shape)
    # print("\n")

    return (B_hat, A_hat)

def main():
    rows = 256
    cols = 4096
    p = 2
    numpy = False

    ranks = []
    accs = []

    if(numpy):
        np.random.seed(42)
        mat = np.random.randn(rows, cols)
    else:
        torch.random.manual_seed(42)
        mat = torch.randn([rows, cols]).cuda()

    for i in range(rows):
        B_hat, A_hat = rand_svd(mat, i, p)
        approx = B_hat @ A_hat

        if(numpy):
            # l1_norm_diff = np.sum(np.abs(mat - approx))
            acc = np.linalg.norm(mat - approx, 'fro')
        else:
            # l1_norm_diff = torch.sum(torch.abs(mat - approx))
            acc = torch.linalg.norm(mat - approx, ord='fro')
        
        print(acc)
        ranks.append(i)
        accs.append(acc.item())

    for r, a in zip(ranks, accs):
        print(f"rank {r}: frob_acc {a}")



if __name__ == "__main__":
    main()