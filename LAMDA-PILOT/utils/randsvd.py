
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