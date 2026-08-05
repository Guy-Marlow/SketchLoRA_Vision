# CE Step/Boundary Isolation Plan (2026-08-04)

Fixes the measured-CE pipeline's real problem: `torch.profiler` sessions are opened
around far more code than needed, so wall-clock cost scales with the *surrounding*
training loop, not with the *overhead* being measured. Supersedes nothing in
`docs/ce_profiling_implementation_plan.md` (that plan's tags, fairness rules R1-R7,
and R2 baseline-vs-actual mechanism are all still correct and reused here) — this
plan is about **where profiler sessions are opened and how their results are scaled**,
not about replacing measurement with estimation again.

User directive this plan implements: two data points per real boundary (task in the
oracle regime, memory-increment cycle in `bounded_memory`) — **(1)** one-off boundary
bookkeeping cost, measured at *every* boundary, and **(2)** per-step recurring
overhead, measured only on that boundary's epoch 0 and extrapolated by `n_epochs` to
estimate the full intra-task/cycle cost. Not yet implemented — plan only, per
2026-08-04/05 review.

## 0. Why the current pipeline is slow (mechanical fact, not a config problem)

`torch.profiler.profile(with_flops=True)` instruments **every op that executes while
the session is open**, regardless of whether that op sits inside a
`record_function("ce/...")` scope. Tags (`ce_region()`) control what `_harvest()`
*extracts* afterward; they do nothing to limit what the profiler *captures* while
active. So a session's wall-clock cost scales with the code span it wraps, not with
the size of what's being measured.

Today's sessions are too broad:

- `bounded_memory_mixin.py`'s `"step"` session wraps all of `_bounded_train_epoch` for
  epoch 0 of a *sampled* cycle (`ce_profile_every`, default 25) — every batch's full
  ViT forward + backward + `optimizer.step()`, just to extract a few small matmuls.
- `trainer.py`'s oracle `"task"` session wraps the **entire** `incremental_train()` —
  every epoch, not just epoch 0 — and has **no step/boundary split at all**. Its own
  comment already flags this ("far heavier... would add real, non-trivial profiler
  overhead on a full campaign"). This is the worst offender and the direct cause of
  the reported slowness on oracle-regime campaigns (e.g. ImageNet-R 20-task).

Fix: narrow the profiled span down to just the isolated extra computation. Sampling
cadence (`ce_profile_every`) was the previous lever for cost control; with narrow
scoping it mostly stops being necessary (§7).

## 1. Three different *shapes* of overhead — three different measurement techniques

Traced every tagged/untagged call site across all four non-trivial methods. Overhead
does not have one uniform shape:

**(a) Isolated, separately-callable extra code** — O-LoRA `_orth_and_l2()`, TreeLoRA
`tree.step/insert_grad/tree_search/get_loss`, InfLoRA `_init_lora_A`/`_update_dualgpm`,
SketchLoRA `_compress()`/`_run_ca_alignment()`, SketchLoRA-CA's per-step
`_ca_stats.update()`/`_ca_buffer_update()`. Ordinary Python calls, separable from the
base forward/backward. **Technique: a profiler session scoped to just the call.**
The measured cost *is* "overhead vs. SeqLoRA" directly — SeqLoRA never executes this
code, no baseline subtraction needed.

**(b) Extra compute embedded inside the shared forward pass** —
`backbone/vit_lora.py::_lora_delta`'s **fold branch** (`if self._fold_enabled:`),
used by O-LoRA/InfLoRA/TreeLoRA (`frozen_delta` matmul + current-slot matmul).
**Confirmed by direct read: this branch carries no `ce_region` tag at all.** There is
no way to isolate it via tag-based narrowing — it's fused into the same op graph as
the rest of the ViT forward. **Technique: R2 baseline-vs-actual only**
(`measure_baseline_and_actual`, already implemented, already cheap — two single-batch
profiled calls, not per-step). SketchLoRA's own sketch-inclusion cost (non-fold
branch, `shared/lora_delta_slot{}_forward`) *does* carry a tag — but see §4's
forward-only caveat before treating that tag as authoritative.

**(c) A genuine extra full forward pass over the task/chunk's data** — InfLoRA's
`_init_lora_A`/`_update_dualgpm`. Shape-(a) technique applies (isolated call), but the
call itself is expensive by nature (a real second pass over every batch) — that
expense is real, not a scoping artifact, and should not be hidden, only not
compounded by also sitting inside a bigger session.

## 2. Recurrence-period taxonomy — THREE categories, not two (a real bug found here)

Every tagged region was checked for *how often it actually fires*, not assumed from
where it lives in the code:

| Period | Examples | Fires |
|---|---|---|
| **Per-step** | O-LoRA `orth_penalty_matmul`/`orth_l2_norms`; TreeLoRA `tree_insert_grad`/`tree_search_ucb`/`tree_get_loss`/`tree_update_similarity`; CA `ca_class_stats_update`/`ca_reservoir_update` | every batch, every epoch |
| **Per-epoch** | TreeLoRA `tree_search_first_call` — `self.all_grad` is rebuilt on the first `tree_search()` call after `new_epoch_init()` resets it to `None`, and `new_epoch_init` is called once per epoch, not once per task | once per epoch, on that epoch's first relevant call |
| **Per-task/cycle** | O-LoRA `orth_prev_cache_rebuild` — guarded by `task_attr != t`, and `t` (`self._cur_task`) only changes *across* tasks, never across epochs within one task | once per task/cycle, on its first relevant call |

The current pipeline sums **everything** captured during a "step" session into one
`aux_macs_per_step` bucket, divides by `steps_per_epoch`, and later re-multiplies by
`n_epochs * steps_per_epoch`.

- Correct for genuinely per-step costs.
- **Accidentally correct for per-epoch costs, but only if the *entire* epoch 0 is
  profiled**: the one-off cost gets diluted by `1/steps_per_epoch` then exactly
  re-multiplied by `steps_per_epoch * n_epochs`, netting `n_epochs * true_cost` —
  which matches reality only because it really does recur once per epoch.
- **Wrong for per-task costs that happen to fire inside the step loop**: same math
  gives `n_epochs * true_cost` for something that only happens **once, total** — a
  **~`n_epochs`x overcount** (≈20x at `tuned_epoch=20`) for O-LoRA's cache rebuild
  specifically. This is real and structural, not a one-off bug; it will reproduce
  under any fix that only makes the profiler cheaper without addressing
  categorization.

Two consequences:
1. **Do not sub-sample steps within epoch 0** for speed — that would break the
   coincidental correctness of the per-epoch case (TreeLoRA), which currently only
   nets right because the *whole* epoch is profiled.
2. **A third ledger category is required**: per-step (scale by `n_epochs *
   steps_per_epoch`), per-epoch (scale by `n_epochs` only, measured once per epoch —
   not folded into the per-step average), per-task/boundary (charged once, never
   scaled). `utils/ops_ledger.py::OpsLedger.record_unit` needs a new field
   (`per_epoch_macs`, or similar) alongside `aux_macs_per_step`/`boundary_macs`.

## 3. O-LoRA's cache rebuild needs to be *moved*, not just re-measured

`_orth_prev_cache_rebuild` is triggered lazily from inside `_orth_and_l2()` (called
every step) via a per-attention-module guard. Recategorizing its *measurement* isn't
enough while its *trigger* stays interleaved with per-step code — the guard makes it
fire unpredictably relative to whatever narrow window happens to be open.

Recommendation: extract it into its own method (e.g. `_refresh_orth_cache()`), called
explicitly once at task/cycle start (mirrors where InfLoRA's `_init_lora_A` and
TreeLoRA's `tree.end_task()` already live as clean, explicit boundary calls). Same
cached values, same guard semantics (`if t > 0`), computed at a predictable point
instead of lazily. **Safety check (relevant to the "no codepath slowdown" constraint,
§8): this removes a per-step attribute-lookup + int-comparison from every step for
the rest of the task, for every run, profiled or not — a strict improvement, not a
regression.**

## 4. Backward-pass attribution — two structurally different problems, not one

Checked whether forward-only measurement (the natural output of narrow
`record_function` wraps) understates true cost, and found **two distinct reasons**,
requiring different remedies:

**4.1 Structurally entangled backward (shape-b costs)** — SketchLoRA's own slot-0 tag
(`shared/lora_delta_slot0_forward`) exists, but it is **forward-only**. Slot 0's
frozen weights (`requires_grad=False`) don't need a weight-gradient, but autograd
still must backprop **through** slot 0's forward ops to route gradients to whatever
trainable state exists earlier in the graph (the same input `x` also feeds other
paths). Backward cost through a frozen-but-graph-connected slot is real and
**not separable in principle** — there's no "isolated backward" to measure, because
skipping it doesn't correspond to a computation you could actually omit. **This is
exactly the case R2 baseline-vs-actual exists for**: it subtracts two *whole-step*
measurements (`merge=True` actual vs. `merge=False` baseline), which correctly
captures the true backward delta without needing to isolate anything.
Consequence: **R2, not the slot-0 tag, should be the authoritative source for
SketchLoRA's embedded-forward step cost** (same as O-LoRA/InfLoRA/TreeLoRA's untagged
fold branch, §1b). The slot-0/slot-1 tags remain useful as a **forward-only
cross-check** (§5), not as the primary number.

**4.2 Genuinely separable backward, just not currently isolated (shape-a costs)** —
O-LoRA's `orth`/`l2` terms are computed from **leaf, trainable, non-detached**
tensors (`A_t`, `B_t.weight`); `A_prev` is explicitly `.detach()`-ed. `orth`'s
backward is a small, genuinely additional computation through `A_t @ A_prev.T` only —
it does **not** reuse any of CE-loss's own (much larger) backward graph, since `orth`
never touches the network's forward output at all. This backward cost is real,
separable in principle, and simply not measured today (forward-only tags
underestimate it by roughly a typical matmul fwd:bwd ratio). **Same reasoning applies
to TreeLoRA's `tree_get_loss`** (operates on `current_grad`/`all_grad`, not on the
network's forward graph). Two honest options, neither implemented without your
sign-off:
  - (i) measure forward-only, document as a conservative floor; or
  - (ii) add a throwaway isolated backward (`torch.autograd.grad(orth_or_reg_loss,
    [affected leaf params], retain_graph=True)`, discarded, run only during the
    epoch-0 profiled window) to get a true number.

## 5. Cross-check opportunity (SketchLoRA only)

Per §4.1, SketchLoRA is the one method where **both** a tag-based measurement
(`shared/lora_delta_slot0_forward`, forward-only) and R2's forward-only half
(`actual_fwd - baseline_fwd`) exist for the same quantity. They should agree (a
comment already embedded in `backbone/vit_lora.py::_lora_delta` says as much: "both
should read the same value for SketchLoRA"). Recommend computing both during
implementation and asserting/logging their agreement — a free correctness check on
the profiler-attribution machinery itself, not available for O-LoRA/InfLoRA/TreeLoRA
(no tag exists in their fold branch to compare against). If they *don't* agree, that's
a signal of a `record_function`/backward-attribution bug worth chasing before trusting
either number.

## 6. Per-method audit (mechanism validated by direct code read, not assumed)

### 6.1 SeqLoRA / noadapt — the zero baseline
Single drifting adapter (SeqLoRA) or fully frozen backbone + closed-form NCM head
(noadapt); neither has a `_stream_end_chunk` override, an aux loss, or any boundary
hook beyond the shared, universally-paid bookkeeping (`update_fc`, `freeze_to_task`,
`after_task`'s scalar assignment). Both data points are 0 by construction — no
measurement needed, and per the existing fairness rule (R1, `ce_profiling_
implementation_plan.md`), shared bookkeeping every method pays is never charged as
method-specific overhead in the first place.

### 6.2 O-LoRA — soft orthogonality penalty
Mechanism: each task gets a fresh LoRA slot; the new slot's down-projection `A_t` is
pushed to be orthogonal to **every** previous (frozen, `.detach()`-ed) task's `A_s`
via `lamda_1 * sum_{s<t} |A_t @ A_s^T|.sum()`, plus an L2 term on the current slot
(`lamda_2`, confirmed **0.0 in every production config checked**
— `round2_slurm_grid/olora_100mb_s1993.json` — so `orth_l2_norms`'s measured cost is
real compute paid every step for a term that currently contributes nothing to the
loss; keep it separately tagged from `orth_penalty_matmul`, which is the
guarantee-critical term). The orthogonality penalty is genuinely load-bearing:
without it O-LoRA is just additive per-task LoRA with growing rank and no
anti-interference mechanism.

- **Boundary**: `_refresh_orth_cache()` (post-refactor, §3) — wrap at every task/cycle
  start.
- **Step**: R2 (embedded fold-branch matmul, shape-b) **+** narrow wrap around
  `_orth_and_l2()`'s forward (shape-a, §4.2 backward caveat applies), summed. Cost
  genuinely **grows across the run** (`A_prev` is `[t*r, dim]`) — this must be
  measured fresh per task, not once globally; my per-task cadence (§7) captures this
  by construction.

### 6.3 InfLoRA — analytic, interference-free subspace construction (DualGPM)
Mechanism, confirmed by re-reading `_init_lora_A`/`update_DualGPM`: rather than
learning the down-projection `A` by gradient descent, InfLoRA sets it **analytically**
from the SVD of the current task's input covariance (`cur_matrix`, built via a real
extra forward pass with `set_collect(True)`), **projected** to either remove or retain
the subspace already spanned by `feature_list` (DualGPM's growing orthonormal bases).
Only `B` is trained by gradient descent; `A` is frozen once set. This is a hard,
analytic construction, structurally different from O-LoRA's soft penalty — it
constructs non-interference rather than merely discouraging interference.
`update_DualGPM`'s `threshold = (lame-lamb)*cur_task/total_sessions + lamb` requires
the a-priori session count `T` (the qualitative "needs a horizon" story from
`context.md` — a scheduling detail, not its own cost category).

**Correction from my first-pass report**: I previously wrote InfLoRA has "no
per-step aux," which is imprecise. InfLoRA has **no isolated per-step call** (true —
no separate aux loss term), but it **does** run `merge=True` (`train_merge=True`), so
it pays the same untagged fold-branch cost as O-LoRA/TreeLoRA (§1b) — its step-type
data point is **R2 baseline-vs-actual, not "none."**

- **Boundary**: `_init_lora_A` (forward pass + per-module float64 SVD) +
  `_update_dualgpm` (forward pass + `update_DualGPM`'s SVDs, itself growing with
  `feature_list[i]`'s column count `r_i` — the exact mechanism `context.md` already
  identifies as causing InfLoRA's real, measured 36.2s→38.8s/cycle wall-clock climb).
  Both are already clean, separate calls in `incremental_train()` — no refactor
  needed, wrap both at every task.
- **Step**: R2 only.

### 6.4 TreeLoRA — bandit-selected gradient-similarity regularizer
Mechanism, confirmed by reading `utils/kd_tree.py` line by line: replaces O-LoRA's
exhaustive "compare against every prior task" with a **KD-tree** over prior tasks'
accumulated gradient snapshots (recursively split by median cosine-similarity-like
score) plus a **UCB bandit** that samples, per step, exactly **one** prior task per
depth-row (`prev_id_matrix`, via `torch.multinomial`) to regularize against
(`tree_lora_loss`). Important, non-obvious split found while tracing this: the
**actual loss computation** (`get_loss`/`tree_lora_loss`) is `O(1)` in task count
(always exactly one sampled comparison per depth-row, not all `t` prior tasks like
O-LoRA) — but the **bandit's own bookkeeping** (`tree_search`'s UCB statistics,
`self.sim`/`self.num_of_selected`, sized `[task_id, ...]`) genuinely is `O(task_id)`
every step. Both are real, both deserve accurate logging; the "guarantee-critical"
comparison is cheap, the machinery that lets it be smart is not.

`_update_similarity` does 24 sequential `.item()` calls per step (24 GPU→CPU syncs) —
invisible to MAC accounting, correctly why time/sync-count (not just MACs) must be
kept (matches the existing R5 convention).

- **Boundary**: `tree.end_task()` — already a clean, explicit call at `_train()`'s
  tail; wraps `tree_end_task_stack`/`tree_end_task_diff`/`tree_build_recursive`, all
  of which genuinely grow with task count (confirmed ~357MB stack+copy at task_id=484
  in the code's own comments) — profiling this faithfully at every boundary will show
  growing cost over a long run, which is correct, not a scoping artifact.
- **Step**: R2 (fold-branch, shape-b) **+** narrow wrap around
  `tree.step()`/`insert_grad`/`tree_search`/`get_loss` (shape-a). **`tree_search`'s
  `all_grad` rebuild is per-epoch, not per-step or per-task (§2)** — needs the new
  per-epoch ledger category, not the per-step or boundary one.

### 6.5 SketchLoRA (+ CA variants) — randomized-SVD compression, the project's own method
Mechanism: two slots (frozen sketch, trainable residual); after each
task/period, `_compress()` folds `(sketch + residual)` back to a bounded rank `r̂` via
`rand_svd`/`rand_svd_probe` (randomized SVD — the guarantee-critical operation this
entire method rests on), giving `O(1)` persistent memory regardless of task count.
Verified this claim is actually reflected in the compute, not just the storage: under
`bounded_eviction` + `sketchlora_rank_cap=128` (every completed production run to
date), `composite_rank = prev_rank + residual_total` is bounded by the cap, so
`rand_svd_probe`'s dominant cost (`[d, working_rank+oversampling]` projection against
`[d,d]` delta, `d=768` fixed) stays **bounded across the whole run** — unlike O-LoRA's
`O(t)` orth penalty or TreeLoRA's `O(task_id)` bandit bookkeeping. The `nocompress`
ablation deliberately removes this bound (by design, to isolate the cost of bounding
rank at all) — under that config, `_compress()`'s cost genuinely should grow, and
per-task measurement (not a single global snapshot) is what reveals whether it does.

Admission-rule logic (`bounded_eviction`/`floor`'s protected-direction extraction)
decides *which rank* to target — cheap itself (rank-sized threshold comparisons), not
a separate cost category; the expensive part is the SVD it feeds into, already
covered by `fold_rank_select`/`fold_merge_*` tags.

- **Boundary**: `_compress()` (all of it — `fold_composite_build`,
  `fold_rank_select`, `fold_merge_*`, `fold_fd_shrinkage`, `fold_slot_realloc`,
  `fold_residual_reset`; `floor`'s extra QR+SVD when that admission rule is active) +
  `_run_ca_alignment()` when CA is on. Wrap at every task; **fix the pre-existing
  timing bug first** (§6.6) before trusting any SketchLoRA boundary number.
- **Step**: R2 authoritative (§4.1); slot-0/slot-1 tags as a forward-only cross-check
  (§5). CA adds genuine per-step cost: `ca_class_stats_update`/`ca_reservoir_update`,
  narrow-wrapped, straightforward (no per-epoch/per-task hybrid found here — the CA
  reservoir resets once per task in `_ca_reset_reservoir`, called from `_train()`'s
  start, not per epoch).

### 6.6 Confirmed timing bug: oracle-mode R2 probe reads post-fold state
`trainer.py`'s R2 probe runs *after* `incremental_train()` returns. For SketchLoRA,
`_compress()` runs *inside* `_train()`, before `incremental_train()` returns — so the
probe measures the sketch rank that will be used **next** task, not the rank actually
in effect while the steps that just ran were training. `bounded_memory_mixin.py`
already hit and fixed this exact bug (the "R7" fix, snapshotting before
`_stream_end_chunk`); the oracle path never got the equivalent fix. Since `r̂` is
constant for a whole task (changes only at the fold), the fix is to snapshot before
`_compress()` fires, not after — and since it's constant per task, one snapshot per
task suffices, no need to re-measure every epoch.

## 7. Oracle-mode plumbing — no driver-side hook exists today; needs small,
   per-method insertions

`bounded_memory_mixin.py` already has the right shape (`_stream_begin_chunk` /
`_bounded_train_epoch` (epoch-0-gated) / `_stream_end_chunk` → boundary-begin / step /
boundary-end). Oracle mode has nothing equivalent — `trainer.py` calls
`incremental_train()` as one opaque call and (per its own comment) cannot see inside
it without invasively rewriting every method's loop. It doesn't need to: every method
with boundary or step overhead already has that code in separable spots inside its
*own* `_train()`/`incremental_train()`:

| Method | Existing separable call sites (no refactor) | Needs refactor first |
|---|---|---|
| InfLoRA | `_init_lora_A` (before `_train`), `_update_dualgpm` (after) | no |
| SketchLoRA | epoch loop, then conditional `_compress()`+CA | no (only the §6.6 timing fix) |
| TreeLoRA | epoch loop, then `tree.end_task()` | no |
| O-LoRA | epoch loop only | **yes** — §3's cache-rebuild extraction |

Minimal change per method: wrap the existing boundary call(s) in a narrow profiler
session at every task; for methods with per-step aux (O-LoRA, TreeLoRA, SketchLoRA-CA)
wrap *just the isolated aux call* (not the surrounding fwd/bwd/optimizer) inside the
loop that already exists, gated on `epoch == 0`. Recommend a small shared helper
(extend `CEProfileController` with an oracle-friendly variant keyed on `epoch == 0`
rather than `cycle % profile_every == 0`) so this isn't duplicated four times with
subtly different logic.

## 8. Bounded-memory plumbing — refine, don't rebuild

The step/boundary *shape* is already correct here; only the session *scope* needs
narrowing (wrap just the isolated aux call inside `_bounded_train_epoch`'s loop body,
not the whole epoch) and the per-epoch category (§2) needs adding for TreeLoRA. No
change to `_stream_begin_chunk`/`_stream_end_chunk` wrapping (already appropriately
scoped to the boundary functions themselves).

## 9. Safety analysis — confirms no codepath-modification-induced slowdown

Checked every proposed change against "modifications must not slow down normal
(unprofiled) training":

- **O-LoRA cache-rebuild extraction (§3)**: removes a per-step attribute lookup +
  comparison, adds nothing unconditional. Net improvement, every run.
- **Epoch-0 gating checks** (`if epoch == 0: <narrow session> else: <unprofiled
  call>`): the branch condition is a loop-local int comparison, already free relative
  to what exists. When profiling is disabled, the "session" resolves to the existing
  `_NULL_REGION` no-op singleton (`utils/ce_profiler.py`, already how `ce_region()`
  works) — one identity check, no tensor/GPU touch.
- **New boundary-wrap call sites** (`_init_lora_A`/`_update_dualgpm`, `_compress()`,
  `tree.end_task()`, `_refresh_orth_cache()`): each is a `with <maybe-session>:`
  wrapped around a call that **already exists as its own statement** — no logic
  change to training itself, only a context-manager wrap using the already-existing
  `CEProfileSession`/`_NULL_REGION` machinery.
- Slowdown **during** an actively-profiled epoch 0 or boundary event is expected and
  accepted per the user's explicit instruction ("slowdowns from the logging are
  fine") — the constraint is only that epochs 1..N-1 and any non-profiled run stay at
  full speed, which every change above satisfies.

## 10. Validation this design newly enables (not just faster/cheaper CE numbers)

Because boundary cost is now measured **fresh at every task/cycle** rather than
sampled/held, the resulting data directly, empirically tests each method's own
qualitative complexity claim:
- SketchLoRA's boundary cost should stay **flat** across a run under
  `bounded_eviction`+`rank_cap` (the `O(1)` claim), and should **grow** under
  `nocompress` (the ablation designed to remove that bound).
- InfLoRA's boundary cost should **grow** with `feature_list[i]`'s column count —
  already suspected from wall-clock (`context.md`'s 36.2s→38.8s/cycle finding); this
  gives a MAC/time-based confirmation, not just an aggregate wall-clock symptom.
- O-LoRA's step-type cost (`orth_penalty_matmul`) should grow `O(t)`; TreeLoRA's
  bandit bookkeeping (`tree_search_ucb`) should also grow `O(task_id)`, while its
  actual regularization loss (`tree_get_loss`) should stay flat — a real,
  checkable prediction from the algorithm's own design.

## 11. Open decisions

1. **RESOLVED (2026-08-05): forward-only floor**, not throwaway isolated backward,
   for O-LoRA's `orth`/`l2` and TreeLoRA's `tree_get_loss`. Document the reported
   step-type number as a conservative floor (true cost including backward is
   somewhat higher, by roughly a matmul's typical fwd:bwd ratio) rather than adding
   the extra per-step-0 compute a throwaway `torch.autograd.grad` call would cost.
2. **InfLoRA boundary sampling** — still open, see the dedicated explanation given
   2026-08-05 in response to a request to clarify what this question even means;
   not yet answered.
3. **Per-epoch ledger category schema** — still open; naming/schema pending, see the
   dedicated explanation given 2026-08-05 for what problem this solves and how it
   would be implemented.

## 12. Corrections made during self-review (2026-08-05 audit pass)

This document folds in corrections found by re-deriving each method's actual
continual-learning mechanism from source and cross-checking against the original
draft plan, rather than trusting the original draft's categorization at face value:

- InfLoRA's step-type overhead was originally reported as "none" — wrong; it pays
  the same untagged fold-branch cost as O-LoRA/TreeLoRA (§6.3).
- O-LoRA's and TreeLoRA's step-type overhead was under-specified as "the isolated aux
  call" alone — both also pay the R2-only fold-branch cost *in addition to* their
  isolated call; the two are summed, not alternatives (§6.2, §6.4).
- The 2-category (step/boundary) ledger design was insufficient — TreeLoRA's
  `tree_search_first_call` requires a third, per-epoch category to be handled
  correctly without depending on the coincidence that epoch 0 is profiled in full
  (§2).
- SketchLoRA's slot-0 tag was initially treated as an equally-valid alternative to
  R2 — it's forward-only and structurally cannot capture the true (backward-inclusive)
  cost; demoted to a cross-check, R2 is authoritative (§4.1, §5).
