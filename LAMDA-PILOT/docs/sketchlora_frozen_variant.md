# SketchLoRA "Frozen" Variant (Plan A §A5.1/§A5.2)

Implemented 2026-07-25 in `models/sketchlora.py`, as a **separate, opt-in
codepath** — every existing config/run using SketchLoRA is byte-for-byte
unaffected unless it sets one of the three new config keys below. Required by
Plan C (`plan_C_task_agnostic.md` §C1/§C2: "SketchLoRA runs the FROZEN version
only") for any SketchLoRA run under the new task-agnostic setting.

## The three changes, and how to turn each on

| Plan A item | Config key | Default (unchanged behavior) | Frozen-variant value |
|---|---|---|---|
| §A5.1 weight_decay=0 for LoRA | `sketchlora_lora_wd` | `None` → single AdamW group, uniform `weight_decay` (old inline behavior in `models/lora.py::_train`) | `0.0` |
| §A5.1 hard rank cap | `sketchlora_rank_cap` | `None` → unbounded (old behavior) | `128` |
| §A5.2 bounded-eviction admission | `sketchlora_admission` | `"global_eps"` → old rank-selection rule, unchanged | `"bounded_eviction"` |

All three are independent switches; Plan C's frozen variant sets all three
together, but e.g. just setting `sketchlora_rank_cap` alone (keeping the old
`global_eps` admission) is also a valid, separately-testable configuration.

## §A5.1: weight_decay=0 on LoRA parameters only

Mechanism: added a new overridable hook, `_optimizer_param_groups()`, to the
shared base class `models/lora.py::Learner` (used by every LoRA-family method).
Its default implementation returns the exact same flat parameter list that was
previously built inline in `_train()` — so every method that doesn't override
it (SeqLoRA, O-LoRA/TreeLoRA which build their own optimizer inline and never
call this hook, InfLoRA likewise) is completely unaffected. SketchLoRA's
`_train()` reaches this hook because it delegates its core training loop to
`super()._train(train_loader)` (it only adds the pre-freeze and post-train
compression steps around that call) — so it is the one method for which this
hook actually fires. When `sketchlora_lora_wd` is set, SketchLoRA's override
splits trainable parameters into two `optim.AdamW` param groups by name
(`lora_A`/`lora_B` in the parameter name vs. everything else, i.e. the
classifier head), assigning `weight_decay=sketchlora_lora_wd` to the LoRA group
and the ordinary `self.weight_decay` to the head group.

## §A5.1: hard rank cap r̂_max

A plain clamp on the selected rank (`r_hat_t`) inside `_compress()`'s adaptive
(`energy_target` set) branch, applied identically whether `admission_rule` is
`"global_eps"` or `"bounded_eviction"` — the plan is explicit that the cap
"lands now (no supervisor dependency)," independent of the admission-rule
decision, which does need sign-off. Cost at d=768: ≈19 MB fp32 across q/v
projections at the cap, per the plan's own estimate.

## §A5.2: bounded-eviction admission rule

**The problem it fixes.** The existing (`global_eps`) rule recomputes the
target rank fresh from the *merged* (sketch ⊕ residual) spectrum's own energy
distribution at every single merge, with no memory of what the previous rank
was. If that merged spectrum's energy happens to concentrate differently after
folding in a particular residual, the freshly-computed target rank can be
**smaller** than the rank the sketch held going into the merge — a sudden,
large rank drop ("retroactive mass-eviction" / the observed post-peak rank
collapse) that discards more history than the residual itself ever
contributed.

**The fix.** At each merge of sketch (rank `R`, the previous compressed rank)
with the residual(s) (combined rank `residual_total`, generically `= lora_rank`
for the default `svd_period=1` case Plan C uses), compute `composite_rank = R +
residual_total` and `k_eps` (the rank the energy threshold would select if
applied with no memory, i.e. exactly what `global_eps` already computes). Then:

- **Below the cap** (`composite_rank <= rank_cap`, or no cap set): the number of
  directions evicted is `evict = min(residual_total, max(0, composite_rank -
  k_eps))` — i.e. **never evict more than the residual's own just-added rank**.
  New rank `= composite_rank - evict >= R` always (monotone non-decreasing).
  Worst case (the threshold wants to cut aggressively): `evict = residual_total`
  exactly, new rank `= R` (flat, not shrunk).
- **At/above the cap** (`composite_rank > rank_cap`): override — evict exactly
  `composite_rank - rank_cap` directions, i.e. truncate exactly enough to
  return to `rank_cap`, regardless of what `k_eps` requested. This is the only
  way rank can ever decrease under this rule, and it only ever returns to the
  cap, never below it.

**Ambiguity flag (unresolved, documented per instruction to flag genuine
uncertainty rather than silently pick one reading):** the plan text is "truncate
`t = min(10, k_eps)` trailing directions... where `k_eps` is the count the eps
threshold requests." Two readings exist:
1. **(Implemented.)** `k_eps` = the threshold's requested **keep**-rank (exactly
   what `global_eps` already computes), and the eviction count is `min(residual
   _total, composite_rank - k_eps)` — i.e. `k_eps` is used to derive how much to
   evict, not read directly as the eviction count.
2. **(Not implemented.)** `t = min(10, k_eps)` literally, i.e. the eviction
   count itself is capped at `min(10, k_eps)`.

Reading 2 produces a **perverse** result: if the threshold wants to cut
aggressively (small `k_eps`), reading 2 evicts *fewer* directions the more
aggressive the threshold is, which contradicts both "rank is monotone
non-decreasing" (stated two lines later in the plan) and the entire stated
purpose (bounding a *mass*-eviction, which requires evicting *more* than
`k_eps` would ask when the naive rule wants to overshoot, not less). Reading 1
is the only one of the two consistent with the plan's own stated invariant and
purpose, so it is what's implemented — flagging this explicitly since the
literal wording supports reading 2 on a first pass, and this should be
re-checked against original intent if a supervisor is available before Plan C
§C-Step 2 results are taken as final.

## Verification

Smoke test (`/tmp/.../scratchpad/smoke_frozen_sketchlora.py`, not committed —
throwaway): CIFAR-100, 5 tasks, 2 epochs/task, `svd_rank=10`,
`svd_energy_target=0.01`, `rank_cap=24` (deliberately low to actually exercise
the cap within 5 tasks). Confirms: no crash under either variant; default
(`global_eps`, no cap, no wd override) reproduces the exact previous behavior
class-for-class; frozen variant (`bounded_eviction`, cap=24, `lora_wd=0.0`)
tracks the same rank trajectory as default while `composite_rank <= 24` (both
reach rank 17 after task 1, confirming the two formulas agree whenever the
naive rule's eviction request is small), then diverges once the cap binds —
full log retained at `run_logs/` is not produced by this throwaway script;
see the printed per-task rank trace in the session transcript for the exact
numbers observed. No cap violation or rank-shrinkage observed in either arm.
