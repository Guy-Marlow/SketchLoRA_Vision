# Plan B — Tuning, Production Grid, Sensitivities, and Reporting (revised)

Prerequisites: Plan A Tier-1 exit criteria green before Wave 1; Tier-2 before
Wave 2. Plan A §A1's frame is normative and not restated. Execution policy: one run
per GPU, no packing, efficiency measured offline from saved checkpoints (Plan A §A3).
Budget numbers in §B7 must be recomputed from Plan A §A2.3's cost table.

Priority order (per PI): (a) Plan A §A2 diagnosis → (b) fair configs for SeqLoRA,
O-LoRA, InfLoRA, TreeLoRA → (c) Wave-1 cluster launch with validated metrics.
SketchLoRA decisions proceed in parallel and nothing about them blocks Wave 1.

---

## B1. Matched training budget

- Epochs: **20/task on ALL five datasets** (adopted at PI direction after the
  throughput fix voided the 10-epoch budget rationale). This matches the PILOT
  convention (EASE/TUNA) and, on Omni-1K, matches the CaRE 100-task protocol
  exactly, making those results directly comparable to the only published
  100-task baselines. Campaign-wide and final: no mixed-epoch tables.
- lr transfer 10→20 ep: no established epoch-scaling rule exists (unlike batch
  scaling); default is peak lr UNCHANGED with the cosine schedule stretched over
  the longer horizon (warmup fixed in absolute steps). Borderline re-sweep calls
  break toward the LOWER arm (longer budgets bias the optimum slightly down).
- Accounting: budget counts TOTAL gradient epochs including auxiliary stages (HiDe
  correction, CL-LoRA init/later, first-session adaptation).
- Batch 48 for every method; native lr linearly rescaled to set sweep centers where
  native batch differed (TUNA 16/32, EASE 16, CL-LoRA 64).
- Native optimizer + schedule per method (SGD/cosine: TUNA, EASE, CL-LoRA, HiDe;
  AdamW/cosine: LoRA-scaffold methods). Optimizer is method design; not unified.

## B2. Learning-rate sweeps (A100s; parallel to Wave 1, not blocking it)

- Wave-1 methods launch tonight AT 20 EPOCHS with their current lr selections
  (peak-lr invariance default); a 20-epoch confirmation re-sweep (4 methods × 2
  datasets × 3 arms, same arm values as before for comparability) runs
  concurrently on the A100s and governs the HELD seeds, not the queued ones
  (§B5 launch procedure). For the re-sweep, ENLARGE the ImageNet-R validation
  split (20% of train, or evaluate over all seen tasks' val data) — the prior
  ~120-img/task split had ±2–4 pt binomial noise, below the decision resolution.
  Pre-registered flip-watch: InfLoRA/IN-R at 1.5e-3 (its justification was
  compensating a 5× epoch cut from native 50; at 20 ep the argument halves) —
  hold one 5e-4 job as insurance.
- Protocol: 3-point sweep {lr₀/3, lr₀, 3·lr₀}, CIFAR-100 + ImageNet-R, 1 seed,
  selected on a validation split (10% of train, stratified, carved before task
  splitting; never test). CIFAR winner → CIFAR/Food101/SUN397/Omni; IN-R winner →
  IN-R; disagreements logged. If a sweep overturns a center, only that method's
  affected jobs are requeued — an acceptable trade under queue contention.
- SketchLoRA sweeps once Tier-2 freezes; the remaining five before Wave 3 (their
  centers are rescaled guesses, so for them sweeps DO block launch).
- Method-specific hyperparameters at paper/repo defaults except §B3. Tuning
  symmetry for the paper: one swept knob (lr) per baseline; lr + ε for SketchLoRA.

## B3. Per-method configurations (sweep centers after batch-48 rescale)

Wave 1:
- SeqLoRA: rank 10, α-null. Center 1e-3 (Food101 5e-4).
- O-LoRA: λ₁ 0.5, λ₂ 0, merging on. Center 1e-3.
- InfLoRA: rank 10, lamb 0.95, lame 1.0; total-task parameter T = true task count
  per split (10/20/20/10/100) — verify the ε_th schedule reads T=100 on Omni.
  Center 5e-4.
- TreeLoRA: reg 0.1, native ViT tree depth; per-task tree statistics logged on
  Omni. Center 1e-3.

Wave 2 (blocked on Plan A Tier-2):
- SketchLoRA (frozen tag): rank 10, α-null, batch 48, period 1, ε 0.01 initial,
  r̂_max 128, bounded-eviction admission (Plan A §A5.2, pending supervisor),
  merge backend per the A5.3 decision (RandSVD default until exact-SVD criterion
  clears), weight_decay 0 on LoRA. Center 1e-3 (Food101 5e-4).
- SketchLoRA+CA (first-class reported variant, per PI): SLCA-style classifier
  alignment — store per-class feature mean + covariance at learning time
  (diagonal variances on Omni-1K: ≈6 MB total vs ≈2.4 GB full); after each task,
  sample pseudo-features from the class Gaussians and fine-tune the linear head
  with CE (head-only, small fixed step budget counted inside the epoch budget).
  Precedent: InfLoRA+CA in the InfLoRA paper — cite it, and note the fairness
  caveat (CA is a bolt-on any baseline could receive) wherever +CA appears.
  SketchLoRA-native alternative (ablation only): SDC-style drift compensation
  using current-task data passed through sketch_old and sketch_new at merge time.

Wave 3:
- CL-LoRA: 10+10 epochs, b48, width 10, split l=6. Center ≈0.0225.
- HiDeLoRA: rank → 10, native α convention, 20 total epochs incl. correction
  (14+6, preserving the main:correction ratio), b48; all inherited auxiliaries
  enumerated explicitly. Center 0.03.
- TUNA: b48, 20 ep, rank 16 native, orthogonality per native per-dataset choice.
  Centers CIFAR 0.03, IN-R 0.06; borrowing stands (Food101←CIFAR, SUN/Omni←IN-R)
  with the SWEPT lr inherited.
- EASE: b48, 20 ep (its native count), native width. Centers CIFAR 0.025, IN-R
  0.15 (aggressive rescale — see §B9.4).
- RainbowPrompt: 20 ep, b48, center 3e-4, native prompt/structural params;
  author-default 100-ep arm is Reserve-only.

## B4. Wave-1 gating: in-flight triage (replaces a blocking pilot for Wave 1 only)

Under queue contention, Wave 1's seed-1993 jobs ARE the pilots. Triage protocol:
as each job's first eval checkpoints land, compare against the published /
prior-analysis ballpark for the MATCHING backbone (IN21K→IN1K columns — see Plan A
§A1 note); a method drifting >3–4 points at comparable progress is killed, fixed,
and requeued before its held seeds release. Do not let a suspect run ride to
completion just because it holds a queue slot. Waves 2–3 revert to the standard
blocking pilot gate (1 seed, CIFAR-100 + IN-R, before their 3-seed launches).

## B5. Production waves — launch procedure (H200 cluster, SLURM)

**Tonight (before the morning rush), after Plan A Tier-1 launch-blocking items:**
1. Submit Wave-1 ACTIVE jobs: {SeqLoRA, O-LoRA, InfLoRA, TreeLoRA} × {CIFAR-100,
   ImageNet-R, Food101, SUN397} × seed 1993 at sweep-center lr (16 jobs), plus
   Omni-1K × seed 1993 × 4 methods (4 jobs). Every sbatch: gpu:1, 12 CPU, 96G,
   resource + canary asserts, state checkpoints on.
2. Submit seeds 1996/1999 for the same grid as HELD jobs (`--hold` or dependency
   on triage sign-off) — they occupy queue position now and release after B4
   triage + sweep confirmation per method.
3. Submit joint-training upper bound (1 seed per dataset) held at low priority.
Requirement: the SUN397 split source (§B9.2) must be pinned before step 1 — it is
now the only decision standing between the plan and the queue.
- **Wave 2 (after Tier-2):** SketchLoRA (and +CA) — same coverage, standard pilot.
- **Wave 3:** CL-LoRA, HiDeLoRA, TUNA, EASE, RainbowPrompt — same coverage,
  blocking sweeps + pilot.
- Omni-1K: seed 1993 for all methods before extra seeds anywhere on Omni; seeds
  1996/1999 for SketchLoRA + anchors {O-LoRA, InfLoRA, TUNA} budget permitting.

## B6. Efficiency measurement (offline, local A100s, from Wave checkpoints)

- Inputs: state checkpoints at t ∈ {1, T/2, T} (Omni {1, 10, 50, 100}) from
  production runs.
- Measured per method per t, batch 1 and 64: forward-pass count and FLOPs (shared
  counting harness, eval configuration). Expected shapes to sanity-check against
  (deviations investigated; MEASURED value reported): SeqLoRA / SketchLoRA /
  O-LoRA(merged) / InfLoRA(folded): 1 forward, O(1) in t (record SketchLoRA r̂
  alongside); TreeLoRA: 1 forward + lookup; CL-LoRA: specific blocks grow with t;
  HiDeLoRA: 2 forwards + statistics classifier; TUNA: (t+1) forwards; EASE:
  per-adapter extraction grows with t; RainbowPrompt: determine empirically.
- Latency: A100, exclusive, 500 warm-up + 2000 timed images, median + p90 — the
  comparative claim rests on identical hardware for all methods; H200 confirmation
  deferred until post-rebuttal slots exist (nice-to-have, not blocking).
- Persistent memory: M_inf(t), M_train(t) byte counts from the serialized
  checkpoints (inclusion lists per Plan A §A4.5).
- From production in-run timers: per-task boundary_ops and train-step-time-vs-t
  (exposes O(t) growth, e.g., O-LoRA projection), wall-clock secondary, node
  context logged.

## B7. Budget and contingency ladder (recompute from Plan A §A2.3)

The pre-fix 3 h anchor is void (SLURM CPU/RAM starvation — Plan A §A2.1); all
numbers below are placeholders until the post-fix SUN397 anchor rerun re-prices
them, and the first canary job's measured img/s finalizes the H200 column. The
20-epoch adoption roughly doubles short-run cost and adds ~⅓ to Omni versus the
old placeholders — recompute before releasing held seeds; post-fix throughput
should still leave ample reserve.
Placeholder planning (~1.2 h/short run, ~10 h/Omni run, unpacked): Wave-1 short
grid 4×4×3 ≈ 58 h; Wave-2 (+CA doubles SketchLoRA) ≈ 29 h; Wave-3 short grid
5×4×3 ≈ 72 h; Omni seed 1993 all ≈ 100 h; Omni extra seeds ≈ 80 h; joint bounds ≈
10 h; total ≈ 350 h committed, remainder reserve. If the fix lands as expected,
the ladder below may not trigger — in that case spend the surplus per the Reserve
priority list (§B8), Omni extra seeds first. Wall-clock = GPU-h / 2 (two H200s),
subject to rebuttal-period queue contention — expect elapsed time to exceed
wall-clock arithmetic; front-load Wave 1 accordingly.
Ladder (apply in order): 1) Omni extra seeds → SketchLoRA only. 2) Short-dataset
seeds: 3 for {SketchLoRA(+CA), SeqLoRA, O-LoRA, InfLoRA, TUNA, EASE}, 1 for the
rest. 3) Omni coverage → {SketchLoRA, SeqLoRA, O-LoRA, InfLoRA, TreeLoRA, best of
TUNA/EASE}. 4) Plan C: C-Step 3 → IN-R at {0.2×, 1×} budgets only, then C-Step 2
→ 1 seed (never cut C-Step 2 entirely — it is the supervisor's headline).
Never cut: O-LoRA/InfLoRA anywhere; Omni seed-1993 down to ≥6 methods; §B6.

## B8. Sensitivities, task-agnostic arm, Reserve

SketchLoRA sensitivities (A100s, during Wave 1; IN-R 20T + Omni 50T, 1 seed/point):
ε ∈ {0.005, 0.0075, 0.01, 0.02} at the current convention (rank-8/α-32 sweeps are
void); r̂_max ∈ {64, 128, ∞}; admission ∈ {bounded eviction (default), global-ε,
guaranteed admission}; period P ∈ {1, 2, 5}; optional FD-shrinkage; CA variant
(SLCA vs SDC-drift). Exact-vs-RandSVD lives in Plan A §A5.3, not here.
Promotion rule: sensitivity points reach the headline table ONLY via validation
selection + 3-seed H200 rerun. Mandatory mechanism logging: admitted-orthogonal-
energy per merge and accuracy-at-arrival (tests the frozen-subspace-saturation
prediction for bounded eviction).

Task-agnostic arm: **superseded by Plan C** (`plan_C_task_agnostic.md`,
bounded-working-memory boundary-free streaming). Per supervisor priority, Plan C
is now the second-highest claim on reserve hours after reruns/debug: C-Steps 0–1
on A100s during Wave 1; C-Step 2 (Omni money experiment) queues after Wave-1
Omni seed-1993 and before Wave-3 Omni extras; C-Step 3 (IN-R budget sweep)
backfills idle capacity. The old Scheme-2 chunk-offset design and the stream_run
smoke harness are retired/demoted per Plan C §C0. i-Blurry/Si-Blurry remain out
of scope (future work).

Reserve priority (revised): reruns/debug; **Plan C Steps 2–3**; RainbowPrompt
author-budget arm (IN-R, 1 seed, 100 ep); Omni seeds beyond ladder; H200 latency
confirmation; TUNA/EASE inference cost at t=100; FD-shrinkage and CA-variant
ablations; Plan C TreeLoRA arm and CIFAR repeat.

## B9. Open decisions

1. **Supervisor sign-offs (block Wave 2 only):** bounded-eviction default
   (A5.2); exact-SVD adoption per the pre-registered criterion (A5.3); +CA as a
   first-class variant (theory-neutral; affects presentation).
2. **Supervisor sign-offs (block Plan C Step 2 only):** the four items in Plan C
   §C8 — lazy-merge admissibility, the Pareto/thin-regime headline framing, the
   exclusion of retired-setting numbers, and the frozen §C6 pre-registration.
3. SUN397 split source — still needed before any SUN397 run (it defines the
   split), but no longer urgent for caching: the pre-resize cache is conditional
   (Plan A §A2.1 step 3) and likely unnecessary after the allocation fix.
4. Pilot promotion: confirmed iff version tags identical.
5. EASE IN-R rescaled center 0.15: accept sweep outcome or clamp the top arm.

## B10. Reporting artifacts

- Compute-honesty table: method × dataset — image-updates, wall hours (node
  context noted), trainable params/task, boundary-ops time.
- Efficiency headline: M_inf(t), FLOPs and forward passes vs t, SketchLoRA r̂
  trajectory overlaid; all-methods-identical-A100 latency table.
- Mechanism figures: SketchLoRA internals (rank, energies, principal angles,
  admitted-orthogonal-energy, accuracy-at-arrival vs t); admission-rule
  comparison.
- Plan C figures: budget dose-response with variance panel; Omni Pareto;
  degradation-from-oracle table (Plan C §C7).
- Accuracy tables: Ā / A_B / forgetting, mean ± std; joint + SeqLoRA bounds in
  every table; budget statement; tuning-fairness statement; +CA fairness caveat.
