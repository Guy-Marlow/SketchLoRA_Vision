# Plan C Execution Log — 2026-07-25 Overnight Session

12-hour / 2-GPU budget (message received ~03:10 NZST, deadline 15:10 NZST).
GPUs 0 and 1 were occupied by another user's jobs all night (`tbai869`,
DCRNN/EvolveODE forecasting runs) and were never touched; all work here ran on
GPUs 2 and 4 only, per the 2-GPU instruction.

## What shipped

1. **C-Step 0.1 — harness archaeology** (`docs/c_step_0_1_harness_archaeology.md`):
   diagnosed the retired `boundary_mult` setting's shared-channel collapse
   (SeqLoRA's own no-bookkeeping control still lost 15-30pts vs its task-
   boundary oracle at matched checkpoints) as most likely caused by the
   global-epoch fold clock desynchronizing from real task completions, with
   eval only firing at folds — confirmed against surviving artifacts, honestly
   flagged as best-effort since the original code no longer exists.
2. **Frozen SketchLoRA** (`docs/sketchlora_frozen_variant.md`,
   `models/sketchlora.py`): weight_decay=0 on LoRA params, hard rank cap
   r_max=128, and a new bounded-eviction admission rule, all as separate,
   off-by-default config flags. Verified via a direct smoke: default variant
   grew unbounded (10→17→23→28→32 over 5 tasks), frozen variant capped exactly
   at 24 and held there (10→17→24→24→24) with no cap violations or rank
   shrinkage. One genuine ambiguity in the plan's own wording is flagged in the
   doc (which reading of "truncate t=min(10,k_eps)" is intended) with the
   reasoning for the reading actually implemented.
3. **Plan C bounded-memory harness** (`models/bounded_memory_mixin.py`,
   `docs/plan_c_bounded_memory_harness.md`): new, additive `boundary_mode:
   "bounded_memory"` codepath -- pre-built full-width classifier head,
   full-logit loss (no local class-window masking), volume-based eval
   checkpoints, InfLoRA's T concession, and a leak audit. Reuses every
   existing method's `_stream_*` bookkeeping hooks unchanged -- no edits to
   olora.py/inflora.py/treelora.py/seqlora.py were needed. Verified end-to-end
   through the real `main.py` entry point, all 4 methods, before any
   production run.
4. **`lazy_merge`** (non-default, `docs/plan_c_bounded_memory_harness.md`):
   implemented as specified, gated off by default per the plan's own
   supervisor sign-off requirement (§C8) -- not used in any run below.
5. **Pre-registration** (`docs/plan_c_preregistration.md`): §C6 committed
   verbatim, dated, before any C-Step 2 run, with notes on tonight's deviations
   (flat-MB budgets, lazy_merge not defaulted, reduced scope).
6. **Budget units changed mid-session** per explicit instruction: flat MB
   (`bm_budget_mb` ∈ {50, 100, ...}) rather than fractions of mean latent task
   size -- documented in the harness doc and every config.

## Runs completed

**C-Step 1 (harness verification, CIFAR-100, 4 tasks, 2 epochs/cycle, 50MB,
seed 1993)** -- all 4 methods passed with no errors:
seqlora 25.35 | olora 24.38 | inflora 24.32 | sketchlora(frozen) 25.10.

**C-Step 0.2 (re-baseline, OmniBenchmark-1K, 15 tasks, 100MB, OLD stream_run
design, seed 1993)** -- frozen SketchLoRA vs the pre-freeze number from
earlier tonight's comparison:

| | pre-freeze (earlier tonight) | frozen (tonight, re-run) | delta |
|---|---|---|---|
| SketchLoRA | 58.59 | **64.07** | **+5.48** |
| (for reference) O-LoRA / InfLoRA | 71.76 / 70.45 | not re-run | gap narrows from ~12-13pts to ~6.4-7.7pts |

**C-Step 2 (the money experiment, OmniBenchmark-1K, 50 tasks, 100MB, full 20
epochs/cycle, seed 1993)** -- all 4 methods, full curves (10 volume
checkpoints, 5%-10%-...-100% of the 84,944-image, 50-task stream):

| checkpoint (% of stream) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| SeqLoRA | 42.76 | 22.47 | 11.09 | 9.98 | 8.31 | 11.85 | 4.74 | 8.53 | 8.27 | **4.99** |
| O-LoRA | 44.52 | 24.58 | 15.23 | 12.81 | 11.21 | 12.91 | 7.91 | 8.22 | 7.92 | **5.87** |
| InfLoRA | 46.11 | 26.50 | 13.31 | 10.61 | 9.46 | 11.51 | 5.84 | 8.25 | 7.87 | **5.69** |
| SketchLoRA (frozen) | 42.26 | 24.21 | 12.81 | 9.05 | 10.33 | 12.45 | 6.11 | 9.21 | 8.68 | **4.91** |

**Bonus (thin-budget check, OmniBenchmark-1K, 15 tasks, 50MB, seed 1993)** --
targeted, apples-to-apples test of §C6 prediction (2) ("thinning budgets:
InfLoRA degrades fastest... crossing below SketchLoRA"), same horizon for both:

| checkpoint (%) | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| InfLoRA | 65.99 | 58.72 | 32.93 | 29.71 | 28.51 | 36.51 | 14.46 | 19.33 | 14.99 | **14.42** |
| SeqLoRA | 62.22 | 50.19 | 24.80 | 25.94 | 23.22 | 22.71 | 14.46 | 19.29 | 11.10 | **8.97** |

InfLoRA leads SeqLoRA at every single checkpoint here (tied only at 70%), by a
growing relative margin toward the end (final: 14.42 vs 8.97, InfLoRA +5.45pts,
+61% relative). This directly **contradicts** §C6 prediction (2) ("thinning
budgets: InfLoRA degrades fastest... crossing below SeqLoRA") -- if anything
InfLoRA is the more robust method at this thinner, shorter-horizon setting.

## Reconciling the two results

The 100MB/50-task run showed near-total convergence (all four methods within
~1pt); the 50MB/15-task run shows InfLoRA cleanly and consistently ahead of
SeqLoRA. Holding budget fixed and only changing horizon isn't possible from
tonight's two data points alone (they differ in BOTH budget and task count),
but the pattern is informative: convergence-to-parity did NOT appear at the
short horizon even with a thinner relative budget (50MB is 0.20x mean task
size vs 100MB's 0.41x) -- it only appeared at the much longer 50-task horizon.
This weakens explanation (a) from below (full-logit loss alone compressing
accuracy across the board regardless of budget) and favors explanation (b):
something about the LONG horizon specifically (500 classes, 122 cycles) is
what drives all methods toward a shared floor, not the full-logit loss or thin
budget in isolation. This is exactly the kind of distinguishing evidence the
original open question called for, though a controlled same-horizon,
same-budget-scaled comparison (e.g. 15 tasks at both 50MB and 100MB, or 50
tasks at 50MB) would be needed to fully isolate the effect -- out of scope for
tonight's remaining time.

## Reading the C-Step 2 result against §C6

**Prediction (1)** ("comfortable budgets... O-LoRA ≥ InfLoRA > SketchLoRA >
SeqLoRA; gaps comparable to the stream-mode smoke") is **not supported** at
100MB/50 tasks: the *direction* of the ordering at the final checkpoint is
right (O-LoRA 5.87 > InfLoRA 5.69 > SeqLoRA 4.99 > SketchLoRA 4.91 -- SketchLoRA
and SeqLoRA essentially tied), but the *magnitude* collapses from the 12-13pt
gap seen in the 15-task/100MB stream-mode smoke down to well under 1 point
across all four methods, and the full curves track within a few points of each
other at every checkpoint, not just the last one. Per the plan's own
falsification clause (§C6 item 4), this is exactly the "crossover structure
does not appear" case -- on this evidence, the mechanism story (subspace
methods meaningfully outperforming on accuracy under task-agnostic streaming)
does not hold at this budget/horizon, and any headline claim should be framed
as the accuracy-resource Pareto only, not an accuracy advantage.

Two candidate explanations for the collapse, not yet disambiguated: (a) the
full-logit loss (no local class-window masking) is a fundamentally harder
training signal than every other setting tested tonight, which may compress
absolute accuracy across the board hard enough to swamp any method-specific
advantage; (b) 50 real tasks (500 classes) is a much longer horizon than the
15-task reference point, and all four methods may simply be approaching a
shared accuracy floor at that scale under this budget regardless of mechanism.
Distinguishing these would need either a full-logit-loss run at the SAME
15-task horizon as the old design (isolating the loss-masking effect) or the
scheduled Omni-1K CIL degradation-from-oracle table (Plan C §C3) -- neither was
in scope tonight given the time budget.
