"""impl_plan_7.28.2026 sec 1 synthetic test: the floor survives the AT-CAP
branch. Constructs a synthetic sketch + residual where the residual has one
direction genuinely NEW (orthogonal to the current sketch) but LOW MAGNITUDE
relative to the sketch's own spectrum -- exactly the case a pure
energy-threshold truncation drops, and exactly the case force_increase's
at-cap branch was found to drop too (2026-07-28, live H200-bound run), since
that branch computed evict = composite_rank - cap with no reference to any
floor at all.

Requires a GPU (utils.randsvd.rand_svd unconditionally calls .cuda() on its
input); no dataset needed -- run with CUDA_VISIBLE_DEVICES set to a free GPU.
"""
import torch

from utils.admission import floor_admission_merge
from utils.randsvd import rand_svd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    raise RuntimeError("rand_svd requires CUDA -- set CUDA_VISIBLE_DEVICES to a free GPU")


def make_scenario(d=20, prev_rank=10, residual_rank=4, weak_sigma=0.05, seed=0):
    """Sketch with LARGE singular values (5..15); a residual whose rank-1
    orthogonal-to-sketch component has a SMALL singular value (weak_sigma) --
    everything else in the residual lies inside the sketch's span (ordinary
    old-direction reinforcement, not new information)."""
    g = torch.Generator().manual_seed(seed)
    U_s, _ = torch.linalg.qr(torch.randn(d, prev_rank, generator=g).to(DEVICE))
    V_s, _ = torch.linalg.qr(torch.randn(d, prev_rank, generator=g).to(DEVICE))
    sigmas_s = torch.linspace(15, 5, prev_rank).to(DEVICE)
    B_s = U_s * sigmas_s.sqrt().unsqueeze(0)
    A_s = (sigmas_s.sqrt().unsqueeze(1) * V_s.t())

    # the one GENUINELY NEW direction: a unit vector orthogonal to span(B_s),
    # given a small singular value so ordinary energy-truncation discards it.
    rand_dir = torch.randn(d, 1, generator=g).to(DEVICE)
    proj = U_s @ (U_s.t() @ rand_dir)
    new_dir = rand_dir - proj
    new_dir = new_dir / new_dir.norm()
    new_term = weak_sigma * (new_dir @ new_dir.t())   # rank-1, orthogonal to sketch, weak

    # remaining residual content: reinforces existing sketch directions (lies
    # IN span(B_s)), moderate magnitude -- ordinary "old task drifts a bit" signal.
    reinforce = 0.3 * (U_s[:, :residual_rank - 1] @ U_s[:, :residual_rank - 1].t())

    R = new_term + reinforce   # this cycle's residual contribution (already [d,d])
    delta_W = B_s @ A_s + R
    return B_s, A_s, R, delta_W, new_dir, weak_sigma


def energy_captured(delta_W, direction, sigma, B_hat, A_hat):
    """How much of the KNOWN weak direction's own energy (sigma^2) survives
    in the reconstruction B_hat @ A_hat, measured by projecting the true
    contribution to delta_W attributable to that direction onto the
    reconstruction's column space span(B_hat)."""
    if B_hat.shape[1] == 0:
        return 0.0
    Q, _ = torch.linalg.qr(B_hat)
    true_contribution = sigma * direction.squeeze(1)   # the weak direction's own vector, scaled
    proj = Q @ (Q.t() @ true_contribution)
    return (proj.norm() / true_contribution.norm()).item() ** 2


def main():
    d, prev_rank, residual_rank = 20, 10, 4
    B_s, A_s, R, delta_W, new_dir, weak_sigma = make_scenario(
        d=d, prev_rank=prev_rank, residual_rank=residual_rank, weak_sigma=0.05, seed=7)

    rank_cap = prev_rank    # composite_rank (prev_rank+residual_rank) > cap -> AT-CAP branch
    energy_target = 0.01
    oversampling = 10

    print(f"=== synthetic floor-survives-at-cap test (d={d}, prev_rank={prev_rank}, "
          f"residual_rank={residual_rank}, rank_cap={rank_cap}) ===")
    print(f"weak new direction singular value = {weak_sigma} vs sketch spectrum in [5, 15]")
    print(f"-> composite_rank={prev_rank + residual_rank} > cap={rank_cap}: AT-CAP branch fires\n")

    # -- k=0 (plain bounded_eviction via floor's own degenerate case) --
    torch.manual_seed(123)
    B0, A0, rank0, _, stats0 = floor_admission_merge(
        delta_W, B_s, R, residual_rank, energy_target, admission_floor_k=0,
        rank_cap=rank_cap, oversampling=oversampling, rand_svd_fn=rand_svd)
    captured0 = energy_captured(delta_W, new_dir, weak_sigma, B0, A0)
    print(f"k=0 (no floor): final_rank={rank0} k_protected={stats0['k_protected']} "
          f"-> weak direction energy retained = {captured0:.4f}")

    # -- k=2 (floor protects it) --
    torch.manual_seed(123)
    B2, A2, rank2, _, stats2 = floor_admission_merge(
        delta_W, B_s, R, residual_rank, energy_target, admission_floor_k=2,
        rank_cap=rank_cap, oversampling=oversampling, rand_svd_fn=rand_svd)
    captured2 = energy_captured(delta_W, new_dir, weak_sigma, B2, A2)
    print(f"k=2 (floor)   : final_rank={rank2} k_protected={stats2['k_protected']} "
          f"-> weak direction energy retained = {captured2:.4f}")

    assert rank0 == rank_cap, f"expected rank clamped to cap={rank_cap}, got {rank0}"
    assert rank2 == rank_cap, f"expected rank clamped to cap={rank_cap}, got {rank2}"
    assert captured0 < 0.5, (
        "expected the weak orthogonal direction to be LARGELY DROPPED without the floor "
        f"(k=0) -- got {captured0:.4f} retained, scenario isn't discriminating enough")
    assert stats2["k_protected"] >= 1, "floor should have protected at least 1 direction"
    assert captured2 > 0.9, (
        "expected the weak orthogonal direction to SURVIVE with the floor (k=2) even "
        f"though composite_rank > rank_cap -- got only {captured2:.4f} retained; the "
        "floor did NOT survive the at-cap branch")

    print()
    print("PASSED: without the floor, the at-cap branch drops the weak-but-genuinely-new "
          "direction (same failure mode found in force_increase); with the floor, the same "
          "direction survives the at-cap branch by construction, at the SAME final rank "
          "(no memory cost difference -- only WHICH directions are kept changes).")


if __name__ == "__main__":
    main()
