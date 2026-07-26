# Plan A — Diagnosis, Gating, and Verification (revised)

Run to completion before the corresponding waves of Plan B. Plans A and B together
supersede `sketchlora_experiment_plan.md`. Gates are now split in two tiers:
**Tier 1 (pre-Wave-1)** — harness-level gates blocking the SeqLoRA / O-LoRA /
InfLoRA / TreeLoRA cluster launch; **Tier 2 (pre-Wave-2)** — SketchLoRA-specific
gates and supervisor decision points, executed in parallel while Wave 1 runs.

Compute: 4 local A100s only. No dedicated H200 validation slots exist (SLURM-only,
contested during the NeurIPS rebuttal window); H200 behavior is validated via the
environment cross-check (A2.2) and the canary protocol (A2.4).

---

## A1. Normative experimental frame (Plan B inherits by reference)

- Backbone: ViT-B/16 pre-trained on ImageNet-21K and fine-tuned on ImageNet-1K
  (the IN21K→IN1K checkpoint), frozen; one backbone for the whole campaign.
  NOTE for the pilot gate: published-number comparisons must use each paper's
  IN1K-variant results where reported (TUNA/EASE report both backbones; most
  PILOT-family tables default to the IN21K-only checkpoint — comparing against the
  wrong column will trip the ±3–4 pt gate spuriously). LoRA methods attach to q
  and v projections only; adapter/prompt methods keep native module types and
  dimensions (parameter counts reported, not equalized).
- Input pipeline, identical for all methods: train = RandomResizedCrop(224) + flip +
  normalize; eval = Resize(256) + CenterCrop(224) + normalize. Transform config
  hashed and logged per run.
- Canonical splits: CIFAR-100 B0 Inc10 (10T); ImageNet-R B0 Inc10 (20T); Food101
  B6 Inc5 (20T, task 1 = 6 classes); SUN397 B37 Inc40 (10T; the exploratory 20T SUN
  split is retired); OmniBenchmark-1K B0 Inc10 (100T).
- Class-order seeds 1993 / 1996 / 1999; identical order across methods within a
  seed; class-order hash logged and asserted equal across methods.
- Evaluation: top-1 over all seen classes, no task oracle, native classifier per
  method. Checkpoint sets (identical across methods): CIFAR-100 and SUN397 every
  task; ImageNet-R and Food101 every 2 tasks; Omni-1K after task 1 then every 10.
- Prediction persistence: per-test-image (image_id, top-1, label) at every eval
  checkpoint; all accuracy metrics computed offline from these artifacts.
- **State checkpointing (new):** full method state dicts saved at t ∈ {1, T/2, T}
  (Omni: {1, 10, 50, 100}). These feed the offline efficiency measurements
  (Plan B §B6); production runs that skip them cannot be used for efficiency claims.

## A2. Slow-run diagnosis and H200 accounting (Priority (a))

### A2.1 Bottleneck: SLURM CPU/RAM starvation — **RESOLVED**
Root cause (confirmed): jobs requested 2 GPUs with default host resources (1 CPU
core, 10 GB RAM) under an 8-worker dataloader — worker thrash starved the GPU.
Fix validated: 2× on CIFAR-100 and 2.4× on JPEG-loading Food101. The optimization
question is closed for launch purposes.
- **Final resource spec (every job, no exceptions):** one job = one run = one GPU
  (`gres=gpu:1`), `--cpus-per-task 12`, `--mem 96G`;
  `num_workers = allocated_cpus − 2` (=10), `pin_memory`, `persistent_workers`,
  `prefetch_factor ≥ 2`.
- Residual monitoring (non-blocking): every run already logs GPU utilization and
  dataloader wait fraction (§A3); if Wave-1 logs show wait fraction >10–15% on
  SUN397/Omni, the conditional pre-resize cache (256 px) is the dormant fallback —
  do not build it preemptively.
- Throughput bonuses (bf16, TF32, channels_last, cudnn.benchmark, torch.compile):
  deferred, non-blocking. Apply only between waves, never mid-wave (a mid-wave
  throughput change is harmless to accuracy but muddies wall-clock comparability;
  if applied, note the boundary in the compute-honesty table).

### A2.2 Environment cross-check (H200 = Hopper, sm_90)
On the cluster environment: (1) `torch.cuda.get_arch_list()` must include sm_90 —
a build without sm_90 binaries silently falls back to PTX JIT compilation, which by
itself could explain the 3 h anchor; (2) CUDA runtime ≥ 12.x and driver version
adequate for it; (3) `torch.cuda.get_device_capability()` returns (9, 0) in-job;
(4) SDPA / attention-kernel availability parity between the A100 and H200 torch
builds (pin identical torch + CUDA versions in both environments where the cluster
permits). Record all four in the run header of every job.

### A2.3 Pricing rule (no dedicated rerun — derive from production logs)
The original 3 h SUN397 anchor is VOID (starved job). No dedicated re-pricing run
is needed: every production job logs img/s and phase times, so the per-run cost
table (short datasets at 10 ep; Omni at 15 ep, eval@10) is assembled from the
first Wave-1 jobs as they complete, and Plan B §B7 is recomputed from it within
the first day of Wave 1 (all rows at the adopted 20-epoch budget; Omni at 20 ep,
eval@10). Planning placeholder until then: post-fix A100
throughput × 1.5 for H200 (conservative, not peak specs).

### A2.4 Canary protocol (replaces dedicated H200 validation)
Every run logs img/s continuously and, at startup, asserts its allocation before
touching data: `SLURM_CPUS_PER_TASK ≥ num_workers + 2` and allocated memory ≥ the
configured floor — fail fast with a clear message if not (this exact starvation
mode must never silently recur; log the full SLURM resource env in every run
header). The first Wave-1 job additionally carries the throughput assertion: if
sustained img/s over the first task falls below the floor (extrapolated H200
estimate × 0.5), dump diagnostics (arch list, JIT status, dataloader wait
fraction, node ID, CPU allocation) and abort before task 2. The first cluster job
IS the H200 validation, and its measured img/s replaces the extrapolation in the
§A2.3 cost table.

## A3. Execution policy (replaces the packing policy)

- One run per GPU, always: each SLURM job requests exactly `gres=gpu:1`,
  `--cpus-per-task 10–12`, `--mem 48–64G` (the A2.1 spec); no process packing, no
  multi-GPU jobs, no default host resources ever again. Wall-clock from any run is
  therefore admissible, with the caveat that SLURM guarantees exclusive GPU, not
  exclusive node: every run logs node ID, allocated CPUs/RAM, and mean dataloader
  wait fraction; wall-clock is reported as secondary to counted metrics (forward
  passes, FLOPs, bytes) in all efficiency claims.
- Efficiency measurement moves OFFLINE: FLOP counting, forward-pass counting, and
  latency micro-benchmarks run on the local A100s against the saved checkpoints
  (A1), eliminating any duplicate-run cost. Only the near-free in-run timers
  (boundary_ops, train-step-time-vs-t) remain in production jobs. H200 latency
  confirmation is deferred until slots free up post-rebuttal; A100 numbers carry the
  comparative claim (identical hardware across all methods).

## A4. Tier-1 gates — harness (restructured for tonight's launch)

Because every reported metric derives from persisted artifacts (predictions, state
checkpoints, logs), the only items that BLOCK queueing are those affecting what
gets persisted. Everything else validates post-queue without wasting the runs.

**A4-blocking (complete before tonight's submission):**
1. Cross-method asserts, exercised on ≥2 methods: transform-hash equality;
   class-order-hash equality; eval-checkpoint-schedule equality; head-growth audit
   (head width == classes seen at every checkpoint).
2. Prediction-persistence round-trip: offline Ā / A_B / R-matrix recomputation
   matches in-run numbers exactly; in-run numbers demoted to logging only.
3. Eval-routing identity: every eval checkpoint asserts + logs the parameter-state
   hash it evaluates; a test drives checkpoint/task indices apart and verifies
   routing (the historical bug class — non-negotiable even tonight).
4. State checkpointing at the designated t values verified on a 3-task smoke run.
5. Resource + environment asserts live in the launcher (A2.4, A2.2 recorded
   in-header).

**A4-deferred (complete during Wave 1, before any offline analysis):**
6. Instrumentation validation: forward-pass / fvcore FLOP counters vs the analytic
   ViT-B + rank-10 LoRA estimate (~2%); serializer byte counts; phase-timer
   overhead < 2%. (Analysis-side; checkpoints already contain what it needs.)
7. Memory inclusion lists reviewed and human-signed for the four Wave-1 methods
   (SeqLoRA; O-LoRA incl. stored past A/B in M_train; InfLoRA incl. DualGPM bases
   in M_train; TreeLoRA incl. per-task leaf adapters + tree in M_inf). Remaining
   methods' lists block their own waves only.

## A5. Tier-2 gates — SketchLoRA (block Wave 2; run in parallel with Wave 1)

### A5.1 Changes landing now (no supervisor dependency)
- weight_decay = 0 for all LoRA parameters (approved).
- Hard rank cap r̂_max = 128 (per PI direction; {64, 128} remain in the Plan B
  sensitivity sweep). At d=768 the cap costs ≈19 MB fp32 across q/v projections and
  negligible FLOPs; with observed rank ≈91 at 20T it should bind only mid-sequence
  on Omni — the intended behavior.

### A5.2 Default admission rule: **bounded eviction** (PI's proposal; needs
supervisor sign-off before Wave 2)
At each merge of sketch (rank R) + residual (rank ≤ 10): truncate
t = min(10, k_ε) trailing directions of the composite, where k_ε is the count the
ε threshold requests; at the cap, truncate exactly enough to return to r̂_max.
Properties to document: rank is monotone non-decreasing below the cap; retroactive
mass-eviction (the observed post-peak rank collapse) is structurally impossible;
eviction is energy-competitive, so the evicted tail may include old low-energy
directions rather than the newcomer's — this is intended; the residual long-horizon
failure mode is frozen-subspace saturation (evict 10 / admit ~0 orthogonal
directions each merge, plasticity via perturbation of retained directions only).
Theory impact: none — the theory is agnostic to the rank-selection policy; every
admission rule instantiates a rank sequence and the per-merge Eckart–Young bound is
unchanged in form. Diagnostics (always on): admitted-orthogonal-energy per merge;
accuracy-at-arrival per task. Guaranteed admission and pure global-ε become
ablation arms (Plan B §B8).

### A5.3 Exact-SVD validation (gated swap; RandSVD stays default until cleared)
Experiment: CIFAR-100 10T + Omni truncated to 10T, 2 seeds, exact vs RandSVD,
everything else fixed. Pre-registered decision criterion: exact SVD is adopted iff
accuracy is no worse (within seed noise) AND merge wall-time is comparable
(≤2× RandSVD per merge — expect far less at d=768). Expectation to set with the
supervisor: parity + reduced variance, not a significant gain — the
(1 + r̂/(p−1)) inflation only becomes severe at r̂ ≈ 90–130, beyond a 10-task
horizon. Theory note for the supervisor: exact truncated SVD is the Eckart–Young
optimum that Lemma 1 measures RandSVD against, i.e., the p→∞ case already named in
Remark 2(ii); adoption tightens Corollary 3 (constants → 1, deterministic bound,
no (1−δ)^L factor) and requires a one-paragraph remark, not a rewrite. The RandSVD
analysis is retained as the large-d scalability variant either way.

### A5.4 Correctness tests (all must pass regardless of A5.2/A5.3 outcomes)
1. Golden oracle: ε=0, r̂ ≥ K·r reproduces the compression-disabled run end-to-end
   (accuracy-trace equality within float tolerance), CIFAR 3T then 10T.
2. Post-merge equivalence: forward(sketch_new) ≈ forward(sketch_old + residual)
   within the truncation bound, asserted on real activations at every merge.
3. Rank/eviction-rule correctness on synthetic spectra: k_ε computed from EXACT
   total energy (Gram of factors, never a truncated estimate); bounded-eviction
   truncation count correct below/at cap; guaranteed-admission arm preserves the
   top-k orthogonal residual directions (test retained for the ablation arm).
4. Version tag on freeze; every headline run records it; no mixed-version tables.

## A6. Exit criteria

**Tier 1 — launch-blocking (→ queue Wave 1 tonight):**
- [ ] A2.1 final resource spec (gpu:1 / 12 CPU / 96G) in every sbatch script.
- [ ] A2.2 environment cross-check recorded (sm_90 in arch list; no PTX JIT).
- [ ] A2.4 resource + canary assertions live in the launcher.
- [ ] A4-blocking gates 1–5 green.
**Tier 1 — deferred (during Wave 1, before offline analysis):**
- [ ] A4-deferred 6–7 complete; §A2.3 cost table assembled from first job logs;
      Plan B §B7 recomputed.
**Tier 2 (→ Plan B Wave 2 may launch):**
- [ ] A5.1 landed; A5.2 supervisor decision recorded; A5.3 criterion evaluated and
      decision recorded; A5.4 tests green; freeze tagged.
