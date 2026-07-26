# Plan C Pre-Registration (§C6) — Committed Verbatim

**Committed: 2026-07-25**, after C-Step 0 (harness archaeology + frozen
SketchLoRA + bounded-memory harness implementation) and C-Step 1 (CIFAR-100
harness-verification smoke, all 4 methods), before any C-Step 2 run, per
`impl_plan_7.25.2026/plan_C_task_agnostic.md` §C4's explicit instruction:
"Then COMMIT THE PRE-REGISTRATION (§C6, dated) to the repo before any further
run."

The text below is copied verbatim from `plan_C_task_agnostic.md` §C6 as it
stood at commit time — no wording changed, no numbers adjusted after seeing any
C-Step 2 result (none had been run yet at the time of this commit).

---

## C6. Pre-registered predictions (commit verbatim, dated, before C-Step 2)

1. Comfortable budgets (≥1×), short horizon: O-LoRA ≥ InfLoRA > SketchLoRA >
   SeqLoRA; gaps comparable to the stream-mode smoke, modestly narrowed by the
   freeze fixes and lazy merge. (High confidence.)
2. Thinning budgets (0.5× → 0.2×): InfLoRA degrades fastest (forward-looking
   constraint construction from thin exposure), crossing below SketchLoRA and,
   at the extreme, approaching or crossing SeqLoRA; O-LoRA's mean holds longer
   but its seed variance inflates; SketchLoRA's curves are flattest in mean and
   variance. (Moderate confidence in the crossover; high in the ordering of
   degradation rates.)
3. Long horizon (Omni, ~250 cycles at 0.5×): InfLoRA hits a structural capacity
   wall (≈2,500 claimed directions vs d=768) with late-stream accuracy-at-
   arrival collapse (high confidence in degradation, low in timing); O-LoRA
   remains accuracy leader via unbounded state (~hundreds of MB, O(cycles) step
   cost); SketchLoRA holds O(1) memory and flat step time within a few points
   of O-LoRA. Headline claim is the accuracy–resource Pareto, NOT an accuracy
   win.
4. Falsification: if the crossover structure of (2) does not appear, the
   mechanism story is dropped and the paper reports the Pareto claim only; if
   SketchLoRA's gap to O-LoRA at comfortable budgets remains ≥10 pts after the
   freeze + lazy merge, the task-agnostic story is reported as efficiency-only.

---

## Notes on applicability given tonight's actual scope/deviations

- **Budget units**: per explicit user instruction (2026-07-25, mid-session),
  tonight's implementation uses flat MB budgets (`bm_budget_mb` ∈ {50, 75, 100,
  200}) rather than §C2's fractions-of-mean-latent-task-size ({0.2×, 0.5×, 1×,
  2×}). Prediction (1)'s "comfortable budgets (≥1×)" and (2)'s "thinning
  budgets (0.5×→0.2×)" should be read against whichever of the four flat MB
  values ends up representing a comparable per-dataset ratio -- these are
  logged per-run (see `[bounded_mem] budget_mb=... -> ... this budget is N.NNx
  mean task size` in every run's log) so the mapping back to the plan's own
  "×mean-task-size" framing stays traceable even though the config field
  itself is now absolute MB.
- **lazy_merge**: implemented as a non-default, explicitly experimental arm
  (see `docs/plan_c_bounded_memory_harness.md`) since supervisor sign-off
  (§C8) was not obtainable tonight -- predictions above that mention "the
  freeze + lazy merge" should be read as "the freeze" only unless a specific
  run explicitly sets the lazy-merge flag, which is noted per-run if used.
- **Scope**: given the 12-hour/2-GPU ceiling for tonight's work (vs. the
  plan's own ≈90–150 GPU-h total estimate for C-Steps 0–3), C-Step 2 was run
  partially, and C-Step 3 (the ImageNet-R budget sweep) may not have been
  reached at all -- see the final report / `docs/plan_c_execution_log.md` for
  exactly what completed within the window. Falsification criterion (4) can
  only be evaluated against whatever subset of cells actually finished.
