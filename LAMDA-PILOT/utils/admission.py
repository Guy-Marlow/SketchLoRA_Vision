"""Admission rule v2 for SketchLoRA (impl_plan_7.28.2026 sec 1): floor +
cap-turnover. Single codepath replacing the two 2026-07-28-direction ablations
guaranteed_admission and force_increase (both retired -- see models/sketchlora.py's
git history for the retired implementations, _guaranteed_admission_merge and the
force_increase branch of the old eviction-count formula).

Rule: below cap, evict t = min(r_residual, k_eps) trailing directions of the
composite EXCEPT the top-k directions of the residual's component orthogonal
to the pre-merge sketch, which are protected this merge (k = admission_floor_k).
At cap, evict (composite_rank - cap) directions FROM THE NON-PROTECTED SET
ONLY (incumbent turnover) -- the floor survives the cap branch by
CONSTRUCTION, not by a second eviction-count formula that has to independently
remember the floor: the target total rank r_hat_t is computed by the exact
same formula bounded_eviction already uses (unchanged, no floor adjustment),
and the k protected directions are always added ON TOP of an energy-fill
budget of (r_hat_t - k_protected), so they can never be among the evicted
set regardless of how small r_hat_t is forced down to by the cap. This is
the precise bug found in force_increase: its at-cap branch computed
evict = composite_rank - cap with no reference to the floor at all, so once
rank hit rank_cap, force_increase silently collapsed into plain
bounded_eviction. Unit tests: scripts/test_floor_admission_synthetic.py
(floor survives at cap), scripts/test_floor_golden_bitexact.py (k=0, cap=128
== current bounded_eviction, bit-exact via torch.equal).

Protection is per-merge only: the protected directions become ordinary
incumbents next merge (no persistent tagging across merges) -- a multi-merge
protection window is v3 territory (impl_plan_7.28.2026 sec 1, note).
"""
import torch


def bounded_eviction_target_rank(prev_rank, residual_total, k_eps, rank_cap):
    """The unmodified bounded_eviction target-rank formula (models/sketchlora.py's
    existing below-cap/at-cap branches), extracted so floor_admission_merge and
    its tests can call the exact same rule without duplicating it.

    k_eps here is already the CONFORMANT reading (the eviction count the energy
    threshold requests, i.e. max(0, composite_rank - keep_rank) computed by the
    caller) -- NOT the raw keep-rank threshold. Kept as a plain int/int
    function (no torch) so the synthetic tests can exercise it without tensors.
    """
    composite_rank = prev_rank + residual_total
    cap = rank_cap if rank_cap is not None else composite_rank
    if composite_rank > cap:
        evict = composite_rank - cap
    else:
        evict = min(residual_total, k_eps)
    return max(1, composite_rank - evict)


@torch.no_grad()
def floor_admission_merge(delta_W, B_s, residual_products, residual_total,
                           energy_target, admission_floor_k, rank_cap,
                           oversampling, rand_svd_fn):
    """Compute the floor-admission (B_hat, A_hat) for one (layer, projection)
    module. Self-contained: computes its own spectrum for both the target-rank
    decision and the protected-direction extraction.

    Args:
      delta_W: [d, d] float, the full composite (sketch + all residual slots'
        products this merge), UNSCALED (matches the rest of _compress).
      B_s: [d, r_prev] the PRE-merge sketch's B factor (used only for its
        column space, via QR -- values not otherwise read).
      residual_products: [d, d] float, R = sum of this cycle's residual
        slot(s)' own factor products (the NEW content, before folding in).
      residual_total: int, sum of the residual slot(s)' own ranks (their
        Linear width) -- NOT derivable from residual_products' shape (that's
        always [d, d] once summed), so passed explicitly by the caller, which
        already has it for the ordinary bounded_eviction formula.
      energy_target: epsilon for the keep-rank threshold (same meaning as
        adaptive mode elsewhere in sketchlora.py).
      admission_floor_k: k, the number of protected directions requested
        (clamped to what's actually available below).
      rank_cap: r_max or None (None => cap is the composite's own rank, i.e.
        uncapped -- see bounded_eviction_target_rank).
      oversampling: passed through to rand_svd_fn for the energy-fill portion.
      rand_svd_fn: injected (utils.randsvd.rand_svd in production) so tests
        can substitute a smaller/deterministic stand-in.

    Returns (B_hat, A_hat, final_rank, S_full, stats) where S_full is delta_W's
    full singular spectrum (for the caller's diagnostics/sigma_next, matching
    every other admission rule's return convention) and stats has
    k_protected/energy_filled for the diagnostics log.
    """
    d = delta_W.shape[0]
    prev_rank = B_s.shape[1]

    # -- target total rank r_hat_t: the UNCHANGED bounded_eviction formula --
    S_full = torch.linalg.svdvals(delta_W.float())
    total = S_full.pow(2).sum()
    if total > 0:
        cum = torch.cumsum(S_full.pow(2), 0) / total
        keep_rank = int((cum < (1.0 - energy_target)).sum().item()) + 1
    else:
        keep_rank = 1
    keep_rank = max(1, min(keep_rank, d))
    composite_rank = prev_rank + residual_total
    naive_evict = max(0, composite_rank - keep_rank)
    r_hat_t = bounded_eviction_target_rank(prev_rank, residual_total, naive_evict, rank_cap)

    # -- protected directions: top-k of the residual's component orthogonal
    # to the pre-merge sketch's column space --
    k = min(admission_floor_k, residual_total)
    if prev_rank > 0 and B_s.float().abs().sum() > 0:
        Q, _ = torch.linalg.qr(B_s.float())
        R_orth = residual_products - Q @ (Q.t() @ residual_products)
    else:
        R_orth = residual_products
    U_o, S_o, Vh_o = torch.linalg.svd(R_orth)
    k_eff = min(k, int((S_o > 1e-12).sum().item()))
    k_eff = min(k_eff, r_hat_t)   # defensive: pathological rank_cap < k edge case
    if k_eff > 0:
        root_o = S_o[:k_eff].sqrt()
        B_protected = U_o[:, :k_eff] * root_o.unsqueeze(0)
        A_protected = root_o.unsqueeze(1) * Vh_o[:k_eff, :]
        protected_recon = B_protected @ A_protected
    else:
        B_protected = torch.zeros(d, 0, device=residual_products.device, dtype=torch.float32)
        A_protected = torch.zeros(0, d, device=residual_products.device, dtype=torch.float32)
        protected_recon = torch.zeros(d, d, device=residual_products.device, dtype=torch.float32)

    # -- fill the remaining budget by ordinary energy-threshold truncation of
    # what's left once the protected reconstruction is removed. k=0 (or a
    # residual with no orthogonal component left) makes this IDENTICAL to
    # plain bounded_eviction's rand_svd(delta_W, r_hat_t, oversampling) call --
    # same delta_W, same r_hat_t, same oversampling, and no RNG-consuming call
    # happened before it (QR/SVD above are deterministic), so the golden test
    # (k=0) is bit-exact by construction, not by a special-cased shortcut. --
    energy_fill = max(0, r_hat_t - k_eff)
    residual_after_protected = delta_W.float() - protected_recon
    if energy_fill > 0:
        B_energy, A_energy = rand_svd_fn(residual_after_protected, energy_fill, oversampling)
    else:
        B_energy = torch.zeros(d, 0, device=residual_products.device, dtype=torch.float32)
        A_energy = torch.zeros(0, d, device=residual_products.device, dtype=torch.float32)

    B_hat = torch.cat([B_protected, B_energy], dim=1)
    A_hat = torch.cat([A_protected, A_energy], dim=0)
    final_rank = B_hat.shape[1]
    stats = {"k_protected": k_eff, "energy_filled": energy_fill}
    return B_hat, A_hat, final_rank, S_full, stats
