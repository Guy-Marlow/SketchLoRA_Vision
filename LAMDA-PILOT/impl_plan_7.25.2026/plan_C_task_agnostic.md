# Plan C — Task-Agnostic Evaluation: Bounded-Working-Memory Boundary-Free Streaming

Companion to Plans A and B; replaces Plan B §B8's "Scheme 2" arm entirely.
Prerequisites: Plan A Tier-2 (frozen SketchLoRA) for any SketchLoRA run here;
Wave-1 oracle runs (Plan B §B5) serve as the aligned controls.
Supervisor sign-offs required before C-Step 2 are listed in §C8 — they gate the
money experiment, not the engineering.

## C0. Status of the three prior task-agnostic settings (all retired)

1. **boundary_mult / sample mode (2026-07-03):** RETIRED. Epoch-denominated
   boundary clock (budget set by each method's own hyperparameters), sub-single-
   pass exposure per constraint event, 8× class-size confound on IN-R, and an
   undiagnosed shared channel that crushed even SeqLoRA (the no-bookkeeping
   control) by 25–30 pts. Its numbers appear NOWHERE in the paper except, at
   most, as retracted motivation with artifacts named (see C-Step 0.1). Its
   surviving contribution is the hypothesis in §C6.
2. **BudgetStreamManager / budget mode:** RETIRED. Chunk=task redefinition,
   coverage throttling (task count in chunks), compute coupled to byte budget.
3. **stream_run / sample mode (current):** DEMOTED, not retired. Its training
   schedule and eval remain task-aware (epochs iterate real tasks; eval pinned
   to task completions), so it tests bookkeeping-misalignment only. It survives
   as an internal ablation harness, not as "task-agnostic training," and its
   smoke results (O-LoRA/InfLoRA +12–13 over SketchLoRA at 15 Omni tasks)
   stand as the pre-registered short-horizon expectation.

## C1. Setting specification

- **Stream:** latent task structure intact and hidden. Classes grouped into
  contiguous latent tasks in the seeded class order (same orders as the oracle
  runs: seeds 1993/1996/1999), one fixed seeded permutation within each task,
  tasks concatenated. Sharp latent boundaries (Split-benchmark task-free
  convention). No boundary flag, task ID, task count, or per-task epoch
  structure anywhere in the training path.
- **Working memory:** the learner owns a raw-data memory of B bytes
  (converted to an image count at 224×224×3). Images arrive in stream order and
  fill the memory; when full, train E epochs of shuffled minibatches OVER THE
  MEMORY CONTENTS, then flush and refill. The final partial fill trains
  normally. E = 20 (the campaign budget): every image is consumed in exactly one
  cycle and trained 20 epochs there, so TOTAL COMPUTE IS INVARIANT across
  budgets and equal to the oracle runs — the budget governs consolidation
  granularity only. This is the general-CL / bounded-buffer regime (DER++
  desiderata; GDumb-style budgeted raw storage) rather than an arbitrary clock.
- **Classifier head:** pre-built over the FULL label space before training,
  fixed topology (per supervisor allowance; standard in general-CL harnesses).
  Uniform for all methods. Training loss computed over full logits; evaluation
  masks logits to classes seen so far, where "seen" is class-level and
  data-derived (from stream position), never task-derived.
- **Bookkeeping:** default trigger = one consolidation event per memory cycle
  (SketchLoRA merge; O-LoRA new slot + freeze; InfLoRA DualGPM recompute + new
  branch; TreeLoRA new leaf + tree update). Additionally, any method MAY
  self-trigger from internal, boundary-blind statistics (Online-LoRA precedent:
  loss-plateau freezing). SketchLoRA lazy merge (flag `lazy_merge`): accumulate
  the residual across cycles, fold only when residual energy/rank saturates
  (thresholds from internal statistics only) — supervisor sign-off §C8.
- **Method concessions (documented in-paper):** InfLoRA's ε_th schedule needs a
  total-count T; grant T := ceil(stream_images / cycle_images), computable a
  priori. SketchLoRA runs the FROZEN version only (bounded eviction, cap 128,
  wd 0). No other concessions.
- **Evaluation:** boundaries unavailable at test too — eval checkpoints fire on
  DATA VOLUME: every 5% of the stream on short datasets (20 points), every 10%
  on Omni-1K (10 points), plus final. Report anytime accuracy A_AUC over these
  checkpoints (i-Blurry convention) and final accuracy. The harness privately
  maps checkpoints to latent-task progress for analysis figures; the model
  never sees it. TIL is not computed (premise: no task identities exist).
- **Leak audit (blocking):** grep/audit the training path for ANY task-indexed
  control flow (epoch loops over tasks, per-task lr restarts, task-conditioned
  masking). The lr schedule is per-cycle cosine (restart each memory cycle),
  identical for all methods. Eval-routing asserts (Plan A §A4.3) extend to
  volume checkpoints.

## C2. Methods and conditions

- Methods: SeqLoRA (no-bookkeeping control), O-LoRA, InfLoRA, SketchLoRA
  (+SketchLoRA lazy-merge arm if signed off). TreeLoRA optional, reserve-funded.
- Budgets B, as fractions of MEAN LATENT TASK SIZE (pre-specified — no post-hoc
  budget selection): **{0.2×, 0.5×, 1×, 2×}**. The 0.2× arm is the clean
  analogue of the retired setting's stress: thin but fully-trained,
  representatively-sampled consolidation slices.
- Aligned oracle control: the Plan B main-table runs (boundary-aware, same
  seeds, same total compute). All Plan C results are ALSO reported as
  degradation-from-own-oracle.
- Seeds: 2 (1993, 1996); 3 on any cell within 1 pt of a claimed crossover.

## C3. Metrics (pipelines reused from Plans A/B)

- Accuracy at every volume checkpoint; A_AUC; final; degradation-from-oracle.
- **Seed variance as a first-class metric:** per-method spread vs budget (the
  retired setting showed O-LoRA spreads of ~13 pts vs SketchLoRA's ~2 under
  consolidation-noise stress — pre-registered to recur in the 0.2× arm).
- Resource curves vs cycle index: M_inf, M_train (O-LoRA slot bank growth),
  train-step time (O-LoRA orthogonality cost is O(cycles)), SketchLoRA r̂.
- Mechanism logging: SketchLoRA admitted-orthogonal-energy and accuracy-at-
  arrival; InfLoRA residual-subspace dimension per cycle; O-LoRA slot count;
  principal angles between consecutive consolidation events' subspaces.

## C4. Execution steps and budget

- **C-Step 0 (A100, ~1 day, no GPU for 0.1):**
  0.1 Harness archaeology on the retired boundary_mult setting: identify the
      shared channel that crushed SeqLoRA (check loss masking + head growth
      against the boundary clock). Written diagnosis committed to the repo.
  0.2 Confirm which SketchLoRA version produced the stream-mode smoke numbers;
      if pre-freeze, re-run that 15-task Omni smoke on the frozen version
      (re-baselines the 12–13 pt gap before anything else is interpreted).
  0.3 Implement: epochs-over-memory loop; volume checkpoints; pre-built head;
      InfLoRA T concession; lazy_merge behind a flag; leak audit + asserts.
- **C-Step 1 (A100, ~1 day):** CIFAR-100 smoke, all methods, B=0.5×, 1 seed —
  harness verification only. Then COMMIT THE PRE-REGISTRATION (§C6, dated) to
  the repo before any further run.
- **C-Step 2 (H200, the money experiment):** Omni-1K, B=0.5×, 4 methods ×
  1–2 seeds (~50–100 GPU-h at post-fix speeds; same per-run cost as oracle
  Omni). This is where InfLoRA's capacity wall and O-LoRA's resource growth
  either appear or don't.
- **C-Step 3 (H200/A100):** ImageNet-R budget sweep — 4 budgets × 4 methods ×
  2 seeds = 32 runs (~30–45 GPU-h). Produces the crossover and variance
  figures. CIFAR-100 repeat only if reserve allows.
- Total incremental: ≈ 90–150 GPU-h — inside the Plan B reserve.

## C5. Scheduling against Plans A/B

C-Step 0–1 run on A100s during Wave 1 (no cluster contention). C-Step 2 queues
after the Wave-1 Omni seed-1993 jobs and before Wave-3 Omni extras; C-Step 3
short runs backfill H200 idle gaps or run on A100s (accuracy-only cells may run
on A100 — hardware does not affect accuracy; all wall-clock claims stay
H200-exclusive per Plan A §A3).

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

## C7. Reporting

- Headline figure 1: accuracy vs consolidation budget (dose-response, IN-R),
  with the crossover region if present; companion variance-vs-budget panel.
- Headline figure 2: Omni-1K Pareto — final/AUC accuracy vs persistent memory
  and vs step time, one point per method, cycle-indexed resource curves inset.
- Degradation-from-oracle table for every cell.
- Retired settings: at most one paragraph of motivation, artifacts explicitly
  named per C-Step 0.1; no numbers from them in any table.

## C8. Supervisor sign-offs (gate C-Step 2, not C-Steps 0–1)

- [ ] Lazy-merge flag admissible as part of default SketchLoRA (else it runs as
      a labeled variant arm only).
- [ ] Pre-registered framing agreed: Pareto + thin-regime robustness are the
      headline hypotheses; comfortable-budget accuracy win is NOT claimed.
- [ ] Retired-settings numbers excluded from the paper (motivation only).
- [ ] §C6 predictions reviewed and frozen.
