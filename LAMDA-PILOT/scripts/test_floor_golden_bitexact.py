"""impl_plan_7.28.2026 sec 1 golden test: admission_rule="floor" with
admission_floor_k=0 must be BIT-EXACT (torch.equal) to plain bounded_eviction
at the same (prev_rank, residual, energy_target, rank_cap=128) -- k=0 means
zero protected directions, so floor_admission_merge's energy_fill degenerates
to exactly bounded_eviction's own rand_svd(delta_W, r_hat_t, oversampling)
call, with the SAME r_hat_t formula and no RNG-consuming operation (QR/SVD
are deterministic) executed before that call in either path. Requires a GPU
(utils.randsvd.rand_svd unconditionally calls .cuda() on its input); no
dataset needed -- run with CUDA_VISIBLE_DEVICES set to a free GPU.
"""
import torch

from utils.admission import floor_admission_merge, bounded_eviction_target_rank
from utils.randsvd import rand_svd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("rand_svd requires CUDA -- set CUDA_VISIBLE_DEVICES to a free GPU")


def bounded_eviction_reference(delta_W, energy_target, prev_rank, residual_total,
                                rank_cap, oversampling, rng_state):
    """Reproduces models/sketchlora.py's bounded_eviction branch exactly:
    compute k_eps from delta_W's spectrum, get r_hat_t via the shared formula,
    then a single rand_svd(delta_W, r_hat_t, oversampling) call."""
    S = torch.linalg.svdvals(delta_W)
    total = S.pow(2).sum()
    if total > 0:
        cum = torch.cumsum(S.pow(2), 0) / total
        keep_rank = int((cum < (1.0 - energy_target)).sum().item()) + 1
    else:
        keep_rank = 1
    keep_rank = max(1, min(keep_rank, delta_W.shape[0]))
    composite_rank = prev_rank + residual_total
    naive_evict = max(0, composite_rank - keep_rank)
    r_hat_t = bounded_eviction_target_rank(prev_rank, residual_total, naive_evict, rank_cap)

    torch.cuda.set_rng_state(rng_state)
    B_hat, A_hat = rand_svd(delta_W, r_hat_t, oversampling)
    return B_hat, A_hat, r_hat_t


def run_case(name, d, prev_rank, residual_rank, rank_cap, energy_target, seed):
    torch.manual_seed(seed)
    B_s = (torch.randn(d, prev_rank) * 0.1).to(DEVICE)
    A_s = (torch.randn(prev_rank, d) * 0.1).to(DEVICE)
    B_r = (torch.randn(d, residual_rank) * 0.1).to(DEVICE)
    A_r = (torch.randn(residual_rank, d) * 0.1).to(DEVICE)
    delta_W = B_s @ A_s + B_r @ A_r
    R = B_r @ A_r

    # rand_svd's torch.randn(..., device=M.device) draws from the CUDA generator
    # (M is moved to .cuda() internally) -- snapshot/restore THAT state, not the
    # CPU default generator, or the two calls would silently draw different
    # random projections despite looking like "same seed."
    rng_state = torch.cuda.get_rng_state()   # snapshot BEFORE either path consumes it

    B_ref, A_ref, r_hat_ref = bounded_eviction_reference(
        delta_W, energy_target, prev_rank, residual_rank, rank_cap, oversampling=10,
        rng_state=rng_state.clone())

    torch.cuda.set_rng_state(rng_state.clone())
    B_hat, A_hat, final_rank, S, stats = floor_admission_merge(
        delta_W, B_s, R, residual_rank, energy_target, admission_floor_k=0,
        rank_cap=rank_cap, oversampling=10, rand_svd_fn=rand_svd)

    assert stats["k_protected"] == 0, f"{name}: k_protected should be 0 when admission_floor_k=0"
    assert final_rank == r_hat_ref, f"{name}: rank mismatch floor={final_rank} vs bounded_eviction={r_hat_ref}"
    b_exact = torch.equal(B_hat, B_ref)
    a_exact = torch.equal(A_hat, A_ref)
    print(f"{name}: d={d} prev_rank={prev_rank} residual={residual_rank} cap={rank_cap} "
          f"eps={energy_target} -> r_hat={final_rank} (ref={r_hat_ref}) "
          f"B_bitexact={b_exact} A_bitexact={a_exact}")
    assert b_exact and a_exact, f"{name}: floor(k=0) is NOT bit-exact to bounded_eviction"


def main():
    print("=== golden test: floor(admission_floor_k=0) == bounded_eviction, bit-exact ===")
    run_case("below-cap, loose eps", d=32, prev_rank=20, residual_rank=10,
              rank_cap=128, energy_target=0.05, seed=1)
    run_case("below-cap, tight eps", d=32, prev_rank=20, residual_rank=10,
              rank_cap=128, energy_target=0.001, seed=2)
    run_case("at-cap (composite > cap)", d=32, prev_rank=25, residual_rank=10,
              rank_cap=28, energy_target=0.01, seed=3)
    run_case("uncapped (rank_cap=None)", d=32, prev_rank=20, residual_rank=10,
              rank_cap=None, energy_target=0.02, seed=4)
    print()
    print("ALL PASSED: floor(k=0) reduces exactly to bounded_eviction (bit-identical),")
    print("both below cap, at cap, and uncapped -- confirms floor is a strict superset,")
    print("not a behavior change, when the protection floor is disabled.")


if __name__ == "__main__":
    main()
