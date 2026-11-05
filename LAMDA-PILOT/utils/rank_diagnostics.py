"""Effective-rank diagnostic for growing-bank LoRA methods (2026-11-05 user
request, ahead of the InfLoRA-Bank / LoRA-Subtraction(bank) / O-LoRA
comparison campaign).

At the end of each task, for every (block, projection) adapter list, forms
the DENSE summed delta W = sum_{t=0}^{cur_task} B_t @ A_t -- the true,
uncompressed update every CIL forward under merge=True actually applies
(same construction models/inflora.py's old ``_fold_into_base``/
models/sketchlora.py's retained-energy diagnostics already use, just
analyzed here rather than folded into a base weight or compressed) -- takes
its SVD, and reports the EFFECTIVE RANK: the minimum number of singular
values whose cumulative squared magnitude covers >=``energy_target`` (99%
by default) of the total.

This answers "how low-rank does the COMBINED per-task adapter bank actually
end up being," independent of each individual adapter's own fixed
``lora_rank`` -- the point being to compare how the three methods'
different cross-task mechanisms (O-LoRA's orthogonality push between new and
past A's, InfLoRA's DualGPM interference-avoidance, LoRA-Subtraction's DRS
input-covariance projection) end up concentrating or spreading the summed
update's spectral energy. Expectation going in (user, 2026-11-05): O-LoRA's
explicit orthogonality penalty actively pushes each new task's A away from
every past one, so its summed bank should stay closer to full rank; InfLoRA
has no such penalty (DualGPM constrains gradients, not the resulting
directions' mutual orthogonality) so some energy overlap/collapse is
plausible; LoRA-Subtraction has no cross-task mechanism of this kind at all
(DRS is single-task-local, see models/lorasub.py's own docstring), so its
summed bank is expected to be the most redundant/lowest-rank of the three.
This diagnostic is what actually measures that, rather than assuming it.

Opt-in only (``log_bank_rank: true`` in a config) -- a full SVD of a dense
[dim, dim] matrix per (block, projection) is a real, non-trivial cost
(O(dim^3) each, dim=768 here), paid once per task, not per step, but not
free enough to run unconditionally on every campaign.
"""
import logging

import torch


@torch.no_grad()
def effective_rank(delta, energy_target=0.99):
    """delta: dense [out, in] tensor. Returns (r_eff, full_rank, singular_values).

    r_eff is the smallest r such that the top-r singular values' squared sum
    covers >= energy_target of the total squared-singular-value mass (i.e.
    the Frobenius-norm energy of delta's best rank-r approximation, by the
    Eckart-Young theorem) -- NOT a thresholded/truncated reconstruction, the
    exact count. float64 throughout: this is a diagnostic read off a summed
    bank that can itself span a wide dynamic range (many tasks' worth of
    summed low-rank updates), and SVD's accuracy benefits from the extra
    precision at negligible extra cost for a one-off, once-per-task call.
    """
    S = torch.linalg.svdvals(delta.double())
    energy = S ** 2
    total = energy.sum().clamp_min(1e-12)
    cum_ratio = torch.cumsum(energy, dim=0) / total
    r_eff = int((cum_ratio < energy_target).sum().item()) + 1
    r_eff = min(r_eff, S.shape[0])
    return r_eff, S.shape[0], S


@torch.no_grad()
def log_bank_effective_rank(tag, cur_task, deltas_by_block_proj, energy_target=0.99):
    """deltas_by_block_proj: dict {(block_idx, proj_name): dense_delta_tensor}
    (every block x projection's own summed-bank delta, already formed by the
    caller -- this function only runs the SVD/rank-counting and logs a
    summary, it doesn't know how to construct the deltas itself since that
    differs per method's own slot-list attribute names, see each model
    file's own call site).

    Logs the full per-block breakdown (useful for spotting whether a FEW
    layers dominate the collapse vs. a uniform effect across all 12) and a
    mean/min/max summary across every (block, projection) pair. Returns the
    same breakdown dict in case a caller wants to persist it (e.g. into
    persistent_state()'s own breakdown or a separate diagnostics JSON) --
    no caller does that yet, this is pure logging for now.
    """
    full_rank = None
    ranks = {}
    for key, delta in deltas_by_block_proj.items():
        r_eff, r_full, _ = effective_rank(delta, energy_target)
        ranks[key] = r_eff
        full_rank = r_full
    vals = list(ranks.values())
    mean_r = sum(vals) / len(vals)
    per_block_str = ", ".join(
        "{}.{}={}".format(li, proj, r) for (li, proj), r in sorted(ranks.items()))
    logging.info(
        "[{}] task {} summed-bank effective rank ({:.0f}% energy): "
        "mean={:.2f} min={} max={} (full_rank={}) | {}".format(
            tag, cur_task, energy_target * 100, mean_r, min(vals), max(vals),
            full_rank, per_block_str))
    return ranks
