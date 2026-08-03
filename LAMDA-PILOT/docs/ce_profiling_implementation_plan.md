# Measured-CE Instrumentation Plan (2026-08-03)

Replaces the hand-derived analytic formulas in `utils/ce_formulas.py` with direct
`torch.profiler` measurement of each method's own overhead regions.

## 0. Why

`ce_formulas.py`'s own header splits its contents into "CONFIRMED FROM CODE" and
"TAKEN FROM THE PLAN'S OWN STATED MAGNITUDES, not independently re-derived." That
caveat proved load-bearing. Three separate defects were found this session by manual
code re-reading alone — none by measurement, because nothing measures:

1. **InfLoRA, missing base-forward cost** (fixed in code 2026-08-03, untested):
   `_init_lora_A`/`_update_dualgpm` each run a *full extra ViT forward* over the
   chunk; only the incremental covariance bookkeeping was charged. ~4× CE understatement.
2. **InfLoRA, flat DualGPM term** (NOT fixed): `inflora_dualgpm_svd_macs()` returns a
   constant `n_modules * dim**3`. The real cost grows with `feature_list[i]`'s column
   count `r_i`, which increases every cycle — this is exactly what makes InfLoRA's
   measured wall-clock climb from 36.2s to 38.8s/cycle over a run.
3. **TreeLoRA, flat per-step + zero boundary** (NOT fixed): `treelora_aux_macs_per_step`
   is a constant, ignoring `tree_search`'s `O(task_id)` work; and TreeLoRA has **no**
   `_ce_boundary_macs_this_cycle` override at all, so `tree.end_task`'s recursive
   tree build — plausibly its single largest cost — is charged as exactly zero.

The pattern is consistent: hand-derivation misses what nobody thought to look for.
Measurement does not.

## 1. Fairness conventions (established; see also the `ce-instrumentation-conventions` memory)

**The test for every operation: would SeqLoRA also pay this?**

| Rule | Consequence |
|---|---|
| **R1. SeqLoRA is zero by construction** | Base ViT fwd+bwd, optimizer step, dataloading, and the *current trainable slot's* own LoRA projections are shared baseline. Never charged to anyone. |
| **R2. Fix the merged-forward cancellation** | *(see §2 — the most important structural change in this plan)* |
| **R3. Never charge a method for our measurement apparatus** | `sketch_diag` reconstruction, `_bounded_param_hash`, `MetricsLogger`, `_bounded_eval` are excluded. Tagged separately so they are visible but not counted. |
| **R4. Charge only what actually ran** | Every region gated on its config flag; off-path ablation code contributes zero. |
| **R5. Log time *and* MACs** | MACs cannot see GPU→CPU syncs, Python-loop launch overhead, or bandwidth-bound copies — which is most of TreeLoRA's real cost. |
| **R6. Sample, don't profile everything** | Profiling every step perturbs what it measures. |
| **R7. Capture aux state *before* the boundary fires** | Current ledger reads post-fold state (§6.1). |

### 2. R2 in detail — the merged-forward cancellation

This is a pre-existing design flaw the new instrumentation should fix, not inherit.

`CE_i = Ops_fb(Tr_i) · ε / Ops_total(Tr_i)`, where `Ops_fb` is measured **per method on
that method's own routing**, and `Ops_total = n_epochs·(fwd + bwd + aux) + boundary`.

O-LoRA/InfLoRA/TreeLoRA run `merge=True` with folding enabled, so their forward pays an
extra `dim²` matmul per module (`nn.functional.linear(x, frozen_delta)` in
`backbone/vit_lora.py::_lora_delta`) that SeqLoRA never pays. SketchLoRA pays an extra
`2·dim·r̂` for its sketch slot. **All of that currently lands in `Ops_fb`, i.e. in both
the numerator and the denominator, where it cancels to first order.** A method with an
expensive forward and no auxiliary work scores CE ≈ 1.0 and appears free.

**Fix:** measure one *shared* baseline — single slot, `merge=False`, the SeqLoRA
configuration — and use it as the numerator `Ops_fb` for every method. Each method's
own forward cost then appears only in the denominator, where its excess over baseline
is correctly charged. Concretely: at cycle 0, profile both `_fwd_bwd_baseline()`
(forced `task=<current slot>, merge=False`) and `_fwd_bwd_actual()` (the method's real
routing); store both; the difference is a new charged category, `merged_forward_excess`.

This also cleanly answers the "fold the sketch into the backbone weights" question from
this session: under the current scheme that optimization is invisible; under R2 it
shows up exactly where it should.

## 3. Mechanism

### 3.1 Region tagging

Wrap each identified region in `torch.profiler.record_function("<label>")`, read
per-label MACs and wall-clock from `prof.key_averages()`. Labels namespaced
`ce/<method>/<region>`, with `ce/_excluded/<region>` for R3 items.

### 3.2 Sampling cadence (R6)

Profile a full cycle every `ce_profile_every` cycles (default **25**); hold each
region's last measured value constant between samples; record `profiled: true/false`
and `interpolated_from: <cycle_idx>` in each ledger record so downstream analysis can
distinguish measured from held values. Rationale: an ~8 min/cycle run sampling every
25th cycle profiles ~20 of 485 cycles (~4%), enough to fit the growth trends (slot
count, `task_id`, `r̂`, `r_i`) that are the entire point, at negligible perturbation.

Additionally force-profile: cycle 0, cycle 1 (first cycle with a non-trivial history),
and the final cycle.

### 3.3 Differential validation pass (opt-in)

Scoped tags only capture what someone remembered to tag — the exact failure mode that
produced defect #1. So add `--ce_differential_check`, which on profiled cycles also runs
the cycle's steps with the method's aux machinery stubbed out (each method exposes a
`_ce_disable_aux()` context manager) and records the total delta. If
`sum(tagged regions) ≪ differential delta`, a region is missing. This is the same role
the bit-exactness bar plays for the sketchlora/countsketch optimizations that have no
reference implementation.

### 3.4 Ledger schema extension

`OpsLedger.record_unit(...)` keeps its signature; `boundary_macs`/`auxiliary_pass_macs`
already accept itemized dicts. Add parallel `*_seconds` dicts and the sampling metadata.
`compute_ce` unchanged for MACs; add `compute_ce_wallclock` alongside.

---

## 4. Per-method region inventory

Ordered as requested: SketchLoRA → O-LoRA → InfLoRA → TreeLoRA.
**Bold** = currently uncounted or miscounted.

### 4.1 SketchLoRA (`models/sketchlora.py`)

**Per-step (every training step, all 20 epochs):**

| Region | Where | Currently |
|---|---|---|
| `sketch_inclusion` | slot-0 forward contribution, `_lora_delta` sums slots {0,1} | formula `2·d·r̂·tokens·n_modules`; **also inside `Ops_fb`, so it cancels (R2)** |
| **`ca_class_stats_update`** | `_bounded_train_epoch` → `ClassStats.update()` | **UNCOUNTED.** Per-*sample* Python loop with Welford updates; in `shared_full` mode does a `[768,768]` outer product **per sample**. Potentially enormous. Gated on `classifier_alignment`. |
| **`ca_reservoir_update`** | `_ca_buffer_update` | **UNCOUNTED.** Per-sample Python loop + `torch.cat` growth. Gated on `ca_real_mix_frac > 0`. |

**Per boundary (once per cycle, only when a fold fires):**

| Region | Where | Currently |
|---|---|---|
| `fold_composite_build` | `_compress`: `delta_W = B_s@A_s + Σ B_r@A_r` per module | folded into one flat `sketchlora_fold_macs` |
| `fold_rank_select` | `svdvals` / full `svd` for `k_eps` | as above |
| `fold_merge_randsvd` | `rand_svd`: randn, `M@omega`, QR, `Q.t()@M`, SVD | as above; **gesvd retry path unmodelled** |
| **`fold_merge_admission_floor`** | `utils/admission.py::floor_admission_merge` — extra QR + **full `torch.linalg.svd(R_orth)`** + a second `rand_svd` | **UNCOUNTED.** Gated `admission_rule=="floor"`. |
| **`fold_fd_shrinkage`** | `utils/fd.py::apply_fd_shrinkage` | **UNCOUNTED.** Gated `fd_shrinkage`. |
| **`fold_slot_realloc`** | `nn.Linear` construction + `copy_` when rank changes | **UNCOUNTED.** Fires on most adaptive-mode folds. |
| **`fold_residual_reset`** | kaiming/zeros re-init of every residual slot | **UNCOUNTED.** Small but real. |
| `ca_alignment` | `align_head`: `ca_steps` × (fwd+bwd on `fc`) + pseudo-feature sampling | formula counts the fwd/bwd; **`sample_pseudo_features`' per-sample Python loop, the `shared_full` Cholesky, and `build_low_rank_factor_cache`'s per-class SVDs are UNCOUNTED** |
| **`lazy_plateau_check`** | `PlateauTracker.should_fold` → `_residual_products()` builds `[d,d]` per module + norms | **UNCOUNTED.** Gated `lazy_merge=="plateau"`. Runs *every* cycle, not just folds. |
| **`lazy_saturation_check`** | `_lazy_should_fold`: **`torch.linalg.svdvals` per module every cycle** | **UNCOUNTED and expensive.** Gated `lazy_merge=="legacy_saturation"`. |
| `_excluded/sketch_diag` | recon error + norms per module per fold | **must be excluded (R3)** |

**Config note:** the round-2 production runs use `merge_op=randsvd`,
`admission=bounded_eviction`, CA off, FD off, lazy off — so most gated regions
contribute zero there, but all are on live ablation paths.

### 4.2 O-LoRA (`models/olora.py`)

**Per-step:**

| Region | Where | Currently |
|---|---|---|
| `orth_penalty_matmul` | `_orth_and_l2`: `A_t @ A_prev.t()` `[r,d]×[d,t·r]` + abs + sum | formula `r²·d·n_modules·slot_count·3`; **off-by-one: uses `_stream_chunk+1`, actual loop is `range(t)` with `t=_stream_chunk`** |
| **`orth_l2_norms`** | `torch.norm(A_t) + torch.norm(B_t)` × 48 module-projections | **UNCOUNTED** (small) |
| **`orth_penalty_backward`** | autograd through the above | estimated as "~2× forward" inside the same constant; **should be measured, not assumed** |

**Per boundary:**

| Region | Where | Currently |
|---|---|---|
| **`orth_prev_cache_rebuild`** | `torch.cat([A_list[s].weight for s in range(t)])` per module-projection, once per cycle (cache invalidated when `_cur_task` advances) | **UNCOUNTED.** At t=484: `[4840,768]` per module-proj, 24 pairs (12 blocks × {q,v}) ≈ **357 MB of copies per cycle** (CORRECTED 2026-08-03 — this table originally said "≈714MB", off by ~2×; recomputed by hand as `t·rank·dim·4 bytes · n_blocks·n_proj = 484·10·768·4·24 = 356,843,520` bytes). Bandwidth-bound, invisible to MACs — R5 applies. |
| `merged_forward_excess` | `frozen_delta` matmul | **cancels today (R2)** |

### 4.3 InfLoRA (`models/inflora.py`)

**Per-step:** *none.* InfLoRA adds no per-step loss term — all its overhead is at
boundaries. (Confirmed: no `_stream_extra_loss` override, no `_ce_aux_macs_per_step`
override.)

**Per boundary:**

| Region | Where | Currently |
|---|---|---|
| `init_lora_A_forward` | `_init_lora_A`: full extra forward over the chunk with `_collect=True` | base cost **fixed 2026-08-03 (untested)**; split out from bookkeeping here |
| `covariance_accumulate` | `_accumulate_cov`: `bmm(x^T,x)` per module per batch | in `inflora_boundary_macs` |
| `init_lora_A_svd` | per-module `torch.linalg.svd(cur)` in **float64**, plus the `feature_mat @ cur` projection | **UNCOUNTED separately** — dtype matters, f64 SVD is several× f32 |
| `update_dualgpm_forward` | `_update_dualgpm`: second full extra forward | fixed 2026-08-03 (untested) |
| **`update_dualgpm_linalg`** | `update_DualGPM`: 2–3 `torch.linalg.svd` per module, `feature_list[i] @ (feature_list[i].t() @ activation)` | **flat `n_modules·dim³` — ignores `r_i` growth entirely (defect #2).** This is the term that should reproduce the observed 36.2→38.8 s/cycle climb. |
| **`feature_mat_rebuild`** | `feature_mat[i] = feature_list[i] @ feature_list[i].t()`, every cycle, every module | **UNCOUNTED.** `O(r_i·d²)`, grows with `r_i`. |
| **`dualgpm_python_loop`** | the `for ii in range(sval_ratio.shape[0])` rank-selection loops, with per-iteration tensor compares | **UNCOUNTED.** Python-loop/sync cost — R5. |
| `free_folded_slots` | `nn.Identity()` replacement | negligible; tag for completeness |

### 4.4 TreeLoRA (`models/treelora.py`, `utils/kd_tree.py`)

**Per-step:**

| Region | Where | Currently |
|---|---|---|
| **`stacked_A_build`** | `_stacked_A()`: `torch.stack` of 24 reshaped `[7680]` rows = 184,320 floats copied **every step** | **UNCOUNTED** |
| `tree_insert_grad` | `current_grad + grad·frac` over `[24,7680]` | inside the flat constant |
| **`tree_search_first_call`** | first call each epoch: `torch.stack(all_accumulate_grads[:task_id])` → `[task_id,24,7680]`. **At task_id=484 that is ~357 MB allocated and copied, 20× per cycle** | **UNCOUNTED.** Likely dominant. |
| **`tree_search_ucb`** | `sim.clone()`, the UCB bound, `softmax`, 2× `multinomial` on `[task_id,24]` | flat constant; **actually `O(task_id)`** |
| **`tree_update_similarity`** | `_update_similarity`: 24-iteration Python loop, each with `torch.sum(torch.abs(...)).item()` → **24 GPU→CPU syncs per step** | **UNCOUNTED, and MACs cannot see it (R5).** |
| `tree_get_loss` | `tree_lora_loss`: 24-iteration Python loop of `[7680]` dot products | inside the flat constant |

**Per boundary — TreeLoRA has NO `_ce_boundary_macs_this_cycle` override; all of this is charged as exactly zero:**

| Region | Where | Currently |
|---|---|---|
| **`tree_end_task_stack`** | `torch.stack(valid_grads).clone()` → `[task_id,24,7680]` ≈ 357 MB at cycle 484 | **UNCOUNTED** |
| **`tree_end_task_diff`** | `for i in range(N-1,0,-1): grads_tensor[i] -= grads_tensor[i-1]` — **484 sequential kernel launches** at the end of the run | **UNCOUNTED** |
| **`tree_build_recursive`** | `KDTreeNode` recursion to depth 24. Each node: gather, `mean`, `torch.mv`, `torch.median().item()`, and **two Python list comprehensions each calling `.item()` per task index** → potentially tens of thousands of GPU syncs per boundary | **UNCOUNTED. Strongest candidate for TreeLoRA's true dominant cost**, and the reason its measured wall-clock rises (34.4→36.2 s/cycle) while its "CE overhead" reads 0.000008 epochs. |

---

## 5. Implementation order

### Step 1 — Shared scaffolding — **ALREADY WRITTEN (untested), do not redo**

Two files exist and pass a syntax + import + back-compatibility check:

- **`utils/ce_profiler.py`** (new). `ce_region(label)` tag helper returning a
  zero-allocation no-op singleton when inactive; `CEProfileSession` (opens
  `torch.profiler`, installs itself globally, harvests on exit, never raises out of
  `__exit__`); `CEProfileController` (sampling schedule + hold-between-samples +
  provenance); `charged_macs`/`charged_seconds` (enforce R3 by excluding
  `_excluded/…`); `measure_baseline_and_actual` (R2 dual baseline).
  Attribution is **exclusive** — `_walk_exclusive` stops at nested `ce/` scopes, so
  regions may nest safely without double-counting. Records MACs, device seconds, host
  seconds, and `sync_ops` (R5).
- **`utils/ops_ledger.py`** (extended, strictly additive). `record_unit` gained five
  optional kwargs (`measured_step_regions`, `measured_boundary_regions`,
  `baseline_step_macs_fwd/bwd`, `profile_provenance`); omitting them reproduces the
  old behaviour exactly, and every ledger written before today still loads.
  `compute_ce` gained `source=` and `baseline_numerator=` (both defaulting to the
  original behaviour); new `compute_ce_report()` returns all four CE variants plus
  how many cycles were *actually profiled* versus held.

**Deliberately recorded side by side:** each cycle carries both the analytic-formula
numbers and the measured ones. That A/B on identical cycles is the §7 validation
criterion — do not "clean up" the duplication until the measured path has been
confirmed on a live run.

### Step 1b — Wire the driver (`models/bounded_memory_mixin.py`) — NOT STARTED

Deliberately left untouched so the tree stays in a consistent state. Required edits:

1. Import `CEProfileController` / `measure_baseline_and_actual`.
2. Construct the controller next to the `OpsLedger` (~line 372), reading
   `args.get("ce_profile_every", 25)`.
3. Replace the one-time `measure_step_macs` block (~lines 411–433) with
   `measure_baseline_and_actual(...)`, keeping *both* results (R2).
4. `controller.begin_cycle(cycle_idx, is_final=(c_end >= total_images))` before the
   epoch loop.
5. Profile **the first epoch only** of a profiled cycle and commit with
   `scale=1/len(loader)`. Wrapping a whole epoch (rather than K individual steps)
   needs no changes to any `_bounded_train_epoch` override — and it correctly
   captures per-epoch amortised costs such as TreeLoRA's once-per-epoch
   `tree_search` stack rebuild, which per-step sampling would misattribute.
   Trace size is bounded: 8 steps/epoch at 50MB, ~29 at 200MB.
6. **Snapshot `self._ce_aux_macs_per_step()` BEFORE `_stream_end_chunk`** (fixes R7 /
   §6.1), then wrap `_stream_end_chunk(loader)` in the `"boundary"` session.
7. Pass everything through to `record_unit`; switch the end-of-run log to
   `compute_ce_report`.

### Step 2 — SketchLoRA
Most regions and the most gated variants; also the R3 test case (`sketch_diag` must
land under `_excluded/`).

### Step 3 — O-LoRA
Smallest surface; fix the off-by-one (§6.2). **The calibration anchor for the whole
approach**: O-LoRA's formula is the one that *was* independently confirmed against
`_orth_and_l2`, so measured-vs-formula agreement here is what licenses trusting the
measured numbers for the methods whose formulas were never confirmed.

### Step 4 — InfLoRA
Boundary-only (it has no per-step term at all). Must reproduce the observed per-cycle
wall-clock rise.

### Step 5 — TreeLoRA
Largest expected correction; the `end_task` regions have no prior estimate whatsoever.

### Step 6 — Cross-method validation
Differential check (§3.3); confirm SeqLoRA still yields exactly 1.0 under the R2
baseline change; regenerate the 50MB CE table.

## 6. Known issues to fix while in here

1. **Ledger ordering (R7):** `record_unit` is called *after* `_stream_end_chunk`, so
   `_ce_aux_macs_per_step()` reads post-boundary state — SketchLoRA's `r̂` is the
   post-fold value, not what was in force during the cycle's training steps. Snapshot
   aux state before the boundary call.
2. **O-LoRA off-by-one** (§4.2).
3. **`fvcore_cross_check` is dead code** — `fvcore` is not installed; the function
   always returns `None`. Either install it (it would be a genuine independent check on
   the profiler's MAC counts) or delete the pretence.
4. **`rand_svd` prints `Target rank: N` per module per fold** to stdout — 24 lines ×
   485 cycles. Unrelated to CE but it is polluting the logs and costing real I/O.

## 7. Validation criteria

- SeqLoRA measures exactly 0 overhead in every tagged region (definitional check).
- Sum of tagged regions ≈ differential-check delta, per method (§3.3).
- Measured per-cycle overhead trends match the observed wall-clock trends already
  plotted in `run_logs/*/plots/mean_train_time_per_cycle_50mb.png`: flat for SeqLoRA,
  rising for O-LoRA / InfLoRA / TreeLoRA, ~flat for SketchLoRA. **This is the single
  best end-to-end sanity check available** — the wall-clock curve is real measured data
  from completed runs, and any instrumentation that disagrees with its shape is wrong.
- O-LoRA's measured per-step orthogonality cost lands within ~2× of its (independently
  confirmed) analytic formula. A large disagreement means the measurement is wrong;
  agreement calibrates trust for the methods whose formulas were *not* confirmed.

## 8. Status (2026-08-03)

**Steps 1 through 5 are all written.** Every file below passes a syntax check, and
`utils.ce_profiler`/`utils.ops_ledger`/`utils.ca`/`utils.kd_tree`/`backbone.vit_lora`
plus all five `models/*.py` files import cleanly (CPU-only, no CUDA touched). A
synthetic (pure-Python, no-GPU) unit test of the controller/ledger interaction also
passed, and caught one real bug in the process (see below) before it could reach a
live run.

**Step 1 — shared scaffolding.** `utils/ce_profiler.py` (new): `ce_region()`, `CEProfileSession`,
`CEProfileController` (now backed by `collections.defaultdict` so any `kind` string
works, not just a hardcoded "step"/"boundary" pair), `charged_macs`/`charged_seconds`,
`measure_baseline_and_actual`. `utils/ops_ledger.py` (extended, additive-only):
`record_unit`'s five new optional kwargs; `compute_ce`'s `source=`/`baseline_numerator=`;
new `compute_ce_report()`.

**Step 1b — driver wiring (`models/bounded_memory_mixin.py`), DONE, with one real bug
found and fixed mid-implementation:** the initial wiring called
`ce_profile_controller.begin_cycle(...)` *after* `_stream_begin_chunk(loader)` — but
`_stream_begin_chunk` is exactly where InfLoRA's `_init_lora_A` (a substantial,
real cost) runs, so any `ce_region()` tags inside it would have been permanently
inert regardless of sampling cadence. Fixed by moving `begin_cycle` earlier and adding
a dedicated `"boundary_begin"` profiling session around `_stream_begin_chunk` (separate
from the pre-existing `"boundary_end"` session around `_stream_end_chunk`), merged into
one `measured_boundary_regions` dict at the `record_unit()` call site. The R7 fix
(snapshot `_ce_aux_macs_per_step()` before the boundary fires), the R2 dual
baseline/actual measurement, and the `compute_ce_report`-based end-of-run log are all
in place as originally planned.

**A second real bug, found via the synthetic unit test**: `compute_ce_report`'s
`n_actually_profiled` counter checked `profile_provenance.get("profiled")` on the
outer dict, but the driver (correctly) passes a dict *of* per-kind provenance dicts
(`{"step": {...}, "boundary_begin": {...}, "boundary_end": {...}}`) — so that check
always read `None`/falsy and **`n_actually_profiled` would have silently reported 0
on every single run, regardless of how many cycles were actually profiled.** Fixed
(`_any_kind_profiled` helper, checks any tracked kind, with a flat-dict back-compat
path) and reconfirmed by the synthetic test.

**Step 2 — SketchLoRA.** All per-step (`ca_class_stats_update`, `ca_reservoir_update`)
and boundary (`fold_composite_build`, `fold_rank_select`, four `fold_merge_*` variants
— split per `merge_op`, one more region than the plan's literal table named — 
`fold_merge_admission_floor`, `fold_fd_shrinkage`, `fold_slot_realloc`,
`fold_residual_reset`, `lazy_plateau_check`, `lazy_saturation_check`, `ca_alignment`,
`ca_logit_adjust` — not in the plan's table, added anyway per R5) regions tagged in
`models/sketchlora.py`; `_excluded/sketch_diag` (×3 call sites) correctly excluded per
R3. `utils/ca.py` carries the two previously-uncounted nested sub-regions
(`ca_pseudo_feature_sampling`, `ca_low_rank_factor_cache_build`) — exclusive
attribution means the outer `ca_alignment` tag reports "align_head minus those two,"
not a grand total; the true total is the sum, exactly what `charged_macs()` already
computes over the whole region dict.

**Step 3 — O-LoRA.** Fixed the off-by-one (`slot_count = self._stream_chunk + 1` →
`self._stream_chunk`, matching `_orth_and_l2`'s actual `range(t)` loop bound — the old
formula charged a nonzero cost at cycle 0, where the real cost is exactly zero).
Tagged `orth_l2_norms`, `orth_prev_cache_rebuild`, `orth_penalty_matmul`. **Two open
issues flagged in code comments, not silently resolved:** (1) `orth_prev_cache_rebuild`
fires once per *cycle* (not once per epoch, unlike TreeLoRA's analogous cache), so
piping it through the standard per-step scale-then-multiply-by-`total_steps` pipeline
overstates its contribution to `Ops_total` by roughly `n_epochs`× (~20×) — no invented
correction applied. (2) whether `record_function` correctly attributes the *backward*
pass back to a forward-tagged region is unverified against torch 2.4.1's actual
behavior — `orth_penalty_matmul` only gives a measured forward floor, not backward.

**Step 4 — InfLoRA.** Tagged `init_lora_A_forward`, `init_lora_A_svd`,
`update_dualgpm_forward`, `update_dualgpm_linalg`, `dualgpm_python_loop` (split per the
plan, inside `update_DualGPM`), `feature_mat_rebuild`, `free_folded_slots` (both the
streaming and oracle call sites). One tag (`covariance_accumulate`) lives in the
shared `backbone/vit_lora.py::_accumulate_cov` — safe because that method only ever
runs when `self._collect` is True, which only InfLoRA's code ever sets.

**Step 5 — TreeLoRA.** Tagged `stacked_A_build` (`models/treelora.py`); in
`utils/kd_tree.py`: `tree_insert_grad`, `tree_search_first_call` / `tree_search_ucb`
(split, per the plan, inside `tree_search`), `tree_update_similarity`, `tree_get_loss`,
and — the three regions the analytic formula has NEVER charged anything for, since
TreeLoRA has no `_ce_boundary_macs_this_cycle` override at all —
`tree_end_task_stack`, `tree_end_task_diff`, `tree_build_recursive`, all inside
`end_task` (called from `_stream_end_chunk`, already covered by the `"boundary_end"`
session, no extra driver wiring needed).

**Step 6 (cross-method validation) is NOT started** — correctly so, per the plan: it
needs a live run to do anything.

All of it — written and not-yet-written alike — carries the standing
`*** UNTESTED ***` marking: local GPUs are unavailable (thermal/damage risk) as of
2026-08-03, and none of this has touched a real CUDA profiler trace. Everything above
was checked as far as static tools allow: `ast.parse` on every file, a real Python
import of every changed module, and a synthetic unit test of the controller/ledger
interaction using fabricated (non-torch) profiler results — which is exactly how the
two bugs above were caught, and exactly why this couldn't have been the *last* check
even if it had passed clean. Target validation venue: the H200 cluster, or the
prepared-but-unsubmitted `imagenetr_slurm_grid`. **Before trusting any number this
produces**, confirm section 7's criteria, starting with the cheapest one: SeqLoRA
measures exactly zero in every tagged region.
