# Project Context: SketchLoRA Vision Benchmark

Reinitialization document. Read this in full before running or modifying anything.
Repo root for this document is `LAMDA-PILOT/` inside the git repo
`svd_sketching_vision` (remote: `git@github.com:Guy-Marlow/SketchLoRA_Vision.git`,
branch `main`). Sibling repo `svd_sketching_language` runs the same research
program on NLP tasks; not covered here.

## 1. What this project is

A benchmark harness proposing **SketchLoRA**, a rank-bounded, SVD-compressed LoRA
adapter for pre-trained-model class-incremental learning (CIL), evaluated against
~9 competing parameter-efficient CL methods inside a heavily modified fork of
**LAMDA-PILOT** (arXiv:2309.07117, a PILOT-style CIL toolbox). Backbone is
ViT-B/16 (timm, pretrained IN21K+IN1K-finetuned) throughout; LoRA-family methods
attach to q/v projections of every transformer block.

Three co-existing evaluation regimes, in order of current relevance:

1. **Oracle CIL** (`trainer.py`'s plain per-task loop): ordinary task-incremental
   training, known task boundaries, the conventional CIL protocol. Datasets:
   CIFAR-100, ImageNet-R, SUN397, Food101, OmniBenchmark-1K.
2. **`stream_mixin.py` / `boundary_mode="sample"`**: DEMOTED to an internal
   ablation harness only (tests whether a method's own bookkeeping needs real
   task boundaries; not "task-agnostic training" — eval is still pinned to real
   task completions). Its numbers never appear in the paper except as retracted
   motivation. Do not use for new headline results.
3. **`bounded_memory_mixin.py` / `boundary_mode="bounded_memory"`** (CURRENT,
   Plan C): the learner owns a raw-data buffer of B bytes (`bm_budget_mb`);
   images fill it in stream order; when full, train `tuned_epoch` (E) epochs over
   the buffer, flush, refill. Classifier head pre-built to full width once,
   never grown. Loss is masked cross-entropy over classes present in the current
   cycle only (data-derived mask, never task-derived). Eval fires on DATA VOLUME
   checkpoints (5%/10% of stream), never on task/cycle count. This is the
   general-CL / DER++ / GDumb bounded-buffer regime. **All current/future
   production runs use this harness.**

## 1.5. Two-cluster operational rule (READ BEFORE running or citing ANY number)

Two physically separate clusters are in play, and conflating them has been a
recurring real risk:

- **Local / "testing" cluster**: this host, 4 A100s (indices 0/1/2/4 usable;
  index 3 is a small 4GB DGX card — OOMs immediately if a config targets it,
  avoid). Direct shell/GPU access. **Local results are declared-methodology-
  validation only, NEVER admissible findings** — used for variant ranking,
  bug-finding, and ablation ordering (e.g. every `sketchlora_boltons` CA/
  admission-rule number in §5/§7 above), not for anything that goes in a
  paper/report as a real result.
- **H200 / "experiments" cluster**: 8x H200, SLURM-only, NO shared filesystem
  with this host, and **no direct access from this session** — no `sbatch`/
  `squeue`/`ssh` reachability confirmed from here. The only way results arrive
  is the user manually copying log/JSON files back into this repo (referred
  to as "H200 logs" or "I've pulled the results from the H200s"). These
  copied-back numbers ARE the admissible ones.
- **Other users share this local host's GPUs** — confirmed collisions this
  project has hit: `nway509` (RainbowPrompt jobs, GPU1/GPU4 observed occupied
  for many hours at a stretch), `hwan397`, `tbai869`. **Never kill another
  user's process to free a GPU; never schedule onto a GPU without checking
  `nvidia-smi --query-compute-apps` first for a PID you don't own.** When
  `CUDA_DEVICE_ORDER` is unset, raw `nvidia-smi` GPU indices can silently
  remap (a config's `"device": ["4"]` has landed on the wrong physical card
  before) — always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Do NOT also set
  `CUDA_VISIBLE_DEVICES` when a config already has an absolute `"device": [N]`
  — the two remaps compound and throw `invalid device ordinal`.
- Thermal/GPU-count constraints on the local cluster have been imposed by the
  user before ("only one card for the next five hours") — these are
  time-boxed and explicit; do not assume a GPU budget beyond what was last
  stated, and do not assume a stated time-box has been lifted without asking
  or re-confirming elapsed time.

## 2. Directory map

```
svd_sketching_vision/                  (git root)
  CL-LoRA/ HiDeLoRA/ HiDePet_HiDeLoRA/ InfLoRA/ O-LoRA/ ProgPrompt/
  RainbowPrompt/ TreeLoRA/             reference repos, gitignored, cited/ported
                                       in comments only, NEVER imported
  literature/                          paper PDFs for every method + LoRA_Sketching.pdf
                                       (SketchLoRA's own theory note)
  rand_svd_impl/                       original randsvd prototype (superseded by
                                       LAMDA-PILOT/utils/randsvd.py, vendored copy)
  LAMDA-PILOT/                         <- everything below is relative to here
    models/                           one file per CL method, mixins in lora.py
    backbone/                         ViT variants (vit_lora.py = shared LoRA scaffold)
    utils/                            randsvd, countsketch, admission, lazy, ca, fd,
                                       ce_formulas, ops_ledger, data_manager, metrics_logger
    exps/                             JSON configs, one per run; organized by campaign
                                       (round2_anchor/, round2_grid/, sketchlora_boltons/,
                                       imagenetr_grid/, final_vision/, review/, ...)
    scripts/                          gen_*.py config generators + *_queue.sh /
                                       run_*.sh launchers (see §12)
    run_logs/                         per-run stdout/stderr logs + metrics JSON
                                       (gitignored — `.gitignore:30` at the
                                       svd_sketching_vision repo root)
    docs/                             design/decision docs (harness archaeology,
                                       Plan C harness spec, frozen-variant spec,
                                       round-2 fixes, execution logs)
    impl_plan_7.25.2026/               Plan A (gating/verification), Plan B (tuning/
                                       production), Plan C (task-agnostic) — the
                                       original oracle-campaign plans
    impl_plan_7.26.2026/               bounded_memory_round2_plan.md
    impl_plan_7.27.2026/               sketchlora_boltons_and_ce_metric_plan.md
    impl_plan_7.28.2026/               sketchlora_v2_plan.md (admission floor, CA repair,
                                       InfLoRA positioning)
    EXPERIMENT_SETUP.md                living HP doc for the ORACLE grid (dataset
                                       splits, per-method native-vs-unified HPs)
    BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md   full retired-streaming-design saga
```

## 3. Competitor roster and where their HPs come from

| Method | Code | Native memory growth | HP source used in bounded_memory grids |
|---|---|---|---|
| SeqLoRA | `models/lora.py` (base Learner) | O(1), one drifting adapter, no bookkeeping | lr 3e-4, rank 10, α-null |
| O-LoRA | `models/olora.py` | O(cycles): one frozen slot + orthogonality penalty per boundary | λ₁=0.5, λ₂=0, lr 1e-3 |
| InfLoRA | `models/inflora.py` | O(cycles): DualGPM bases | lamb=lame=0.975 (CONSTANT — see §8, this differs from the oracle grid's asymmetric 0.95/1.0), lr 5e-4; needs a total-session count T, granted as `T := ceil(stream_images/cycle_images)` via `_bounded_set_total_sessions` |
| TreeLoRA | `models/treelora.py`, `utils/kd_tree.py` | O(cycles): hierarchical leaf tree | reg=0.1 (corrected from an inherited reg=0.5 that traced to the NLP/TRACE launch script, not the paper's vision default), lr 1e-3 |
| SketchLoRA | `models/sketchlora.py` | **O(1)**: two slots, sketch+residual | see §4-6 |
| HiDeLoRA | `models/hidelora.py` | — | EXCLUDED from all current work — genuinely broken under any streaming/bounded-memory design (see §10, "HiDeLoRA collapse") |
| ProgPrompt | `models/progprompt.py` | — | status open/unresolved under streaming (sequence-length train/eval mismatch, §10); not in current bounded_memory grids |
| CL-LoRA, TUNA, EASE, RainbowPrompt | native/ported | — | oracle-grid only; not part of the bounded_memory task-agnostic track |

Every method in the current bounded_memory grids runs at **rank=10, lora_alpha=null
(scale ×1), batch=48, tuned_epoch=20** — a deliberately unified convention
(2026-07-28 user directive: "we are doing 1x scaling, with rank 10 adapters, as we
should be doing with every single other test"), superseding an earlier, since-corrected
rank8/α32(×4-scaling) draft. Optimizer/regularizer hyperparameters otherwise stay
each method's own native design (not unified).

**CE-metric audit finding (2026-07-28, InfLoRA — impl_plan_7.28.2026 §4):**
`utils/ops_ledger.py`'s CE formula is `min(1, (1/N) * sum[Ops_fb*eps/Ops_total])`;
InfLoRA's `_ce_boundary_macs_this_cycle` (`models/inflora.py`) charges only the
INCREMENTAL covariance-bookkeeping cost of its two dedicated extra full passes
per cycle (`_init_lora_A` at chunk-begin, `_update_dualgpm` at chunk-end — each
a full forward over the whole chunk to accumulate `cur_matrix`), via
`utils/ce_formulas.py::inflora_boundary_macs` — that function's own docstring
admits its ~16%-of-forward magnitude is "taken from the plan, not independently
re-derived." Auditing the actual call site
(`bounded_memory_mixin.py::bounded_memory_run`, the `ce_ledger.record_unit(...)`
call) confirms `auxiliary_pass_macs` is NEVER populated — meaning the passes'
own BASE forward-pass compute (not just the marginal covariance-accumulation
addition on top of it) is missing from `ops_total` entirely. Back-of-envelope
correction using one real run's numbers (`persistent_state_breakdown`:
frozen_delta=54.0MB fixed, dualgpm_bases 32.9->35.4MB growing, current_slot=
1.4MB, fc~2.9MB; `inference_flops_per_image`=40.9 GFLOPs =~20.45 GMACs): the
missing base-forward term is roughly 7x LARGER than the covariance-bookkeeping
term currently counted, implying InfLoRA's TRUE CE is closer to **~0.96**, not
the reported **~0.993-0.994** (round2_slurm_grid/imagenetr_grid numbers above
all use the UNCORRECTED formula). **Not yet fixed** — needs a user decision
(project-wide ledger fix, changing every historical InfLoRA CE number, vs.
document as a known caveat) before any code changes, per this project's
pre-registration discipline (§9) against post-hoc changes to reported metrics.
Also computed alongside this audit: InfLoRA's best-case-folded persistent
footprint (folding `frozen_delta` into the base weight matrix, valid for
INFERENCE-only deployment) would be 92.1-54.0 = **38.1MB**, vs the
as-implemented 92.1MB — but M_train (what's needed to keep LEARNING future
tasks) still requires the full DualGPM basis footprint regardless, since
those bases have no role at pure inference time but are load-bearing for
projecting future tasks' covariance. Oracle-concession qualitative summary:
InfLoRA is the only method needing the total horizon T a priori (injected via
`_bounded_set_total_sessions`); O-LoRA/TreeLoRA need no horizon but grow state
linearly; SketchLoRA needs neither (no horizon, O(1) state) — a genuine
qualitative differentiator for the frontier story, not just a footnote.

## 4. SketchLoRA mechanism

Two adapter slots on the shared LoRA scaffold (`backbone/vit_lora.py`):
- **Slot 0 (SKETCH)**: frozen, rank-r̂ (`svd_rank`) factor pair (B̂,Â). Zero at init.
- **Slot 1+ (RESIDUAL)**: trainable rank-r (`lora_rank`) factor pair(s); with
  `svd_period=P>1` there are P residual slots cycled round-robin; P=1 (default)
  reduces every formula below to the single-residual case.

Forward (training): sums slots 0..active (`merge=True`) —
`W·x + s·(B̂Â)·x + s·(B_new A_new)·x`. Inference/deployed forward
(`_deployed_forward`): routes through slot 0 ALONE (`task=SKETCH`), bit-exact with
training output because the residual is a mathematically-guaranteed zero-valued
no-op immediately post-compress — this is what gives the paper's claimed O(r̂d)
inference cost instead of training's O((r̂+r)d).

**Compression (`_compress`)**, fired at a task boundary / period boundary / lazy-merge
trigger:
1. `delta_W = B̂Â + Σ B_r A_r` per (block, {q,v}) module (24 modules, 12-block ViT).
2. Choose target rank `r_hat_t` per the admission rule (§5).
3. Factor `delta_W` to rank `r_hat_t` per the merge algorithm (§6).
4. Optionally apply FD shrinkage (§6).
5. Write into slot 0; reset every residual slot (kaiming-A, zero-B).

Persistent memory is **O(1) in task/cycle count** — the entire competitive claim
against every unbounded-memory competitor above.

Every config flag below defaults to exactly reproducing prior behavior when unset —
enforced via bit-exact golden tests (`scripts/test_floor_golden_bitexact.py`,
Round-2 §2.1's ε=0/rank_cap gate).

## 5. Admission-rule lineage (rank selection under adaptive/`energy_target` mode)

Config key `sketchlora_admission`. Three real generations; two intermediate
attempts retired same-day for a confirmed bug.

1. **`global_eps`** (original): recompute target rank fresh from the merged
   spectrum's own energy every merge, no memory of previous rank. BUG: can evict
   more rank than the residual just contributed ("retroactive mass-eviction" /
   post-peak rank collapse).
2. **`bounded_eviction`** (Plan A §A5.2, "frozen" v1 — **used in every completed
   production run to date**: round2_anchor, sketchlora_boltons, imagenetr_grid,
   round2_grid all set `sketchlora_admission=bounded_eviction`,
   `sketchlora_rank_cap=128`, `sketchlora_lora_wd=0.0`): never evict more than the
   residual's own just-added rank per merge → rank monotone non-decreasing below
   the cap. At/above cap: evict exactly `composite_rank - cap`.
   `sketchlora_eviction_reading` has two implemented, unit-tested readings of an
   ambiguous spec sentence (`conformant` = default/production, `literal_keeprank`
   = proven perverse, never used).
3. Two 2026-07-28 attempts to add a "floor" (protect newly-admitted orthogonal
   directions from eviction), both retired same day:
   - `guaranteed_admission`: reserved-slot protection, no floor-survives-cap guarantee.
   - `force_increase`: **confirmed bug** — its at-cap branch computed
     `evict = composite_rank - cap` with no reference to the floor at all, so once
     rank hit `rank_cap` it silently degenerated into plain `bounded_eviction`.
     Evidence on disk: `exps/sketchlora_boltons/50mb_force_increase_k1.json` and
     `50mb_guaranteed_k5.json` were both **killed mid-run** (stopped at cycle 20/73
     of ~75) once this was understood — their `run_logs/.../50mb_force_increase_k1_gpu0.log`
     and `50mb_guaranteed_k5_gpu0.log` have no FINAL line, diagnostic evidence only,
     not usable results. Only the uncapped variant
     (`50mb_force_increase_k1_nocap`/`100mb_force_increase_k1_nocap`, `rank_cap=None`,
     making the at-cap branch structurally unreachable) completed:
     top1/top5 = 65.18/92.37 (50MB), 67.63/92.77 (100MB), Omni-1K 15T.
4. **`floor`** (`utils/admission.py::floor_admission_merge`, CURRENT, replaces both
   retired attempts): protects the top-k directions of the residual's component
   *orthogonal to the pre-merge sketch* (via QR projection onto slot-0's column
   space), sized so the protected k are ADDITIVE to the energy-fill budget
   (`r_hat_t - k`), never competing with it — fixes the bug by construction.
   `admission_floor_k` (default 1) controls k. `k=0` is proven bit-identical to
   plain `bounded_eviction`. **This is implemented in code (uncommitted as of this
   writing — see §11) but has NOT yet been used in any completed production run.**
   Requires `merge_op=="randsvd"` only (not wired for exactsvd/countsketch/naive_sum/nocompress).

## 6. Merge algorithms, lazy-merge, FD shrinkage

**`merge_op`** (`randsvd` default / `exactsvd` / `countsketch` / `naive_sum` /
`nocompress`): ablation ladder. `randsvd` = `utils/randsvd.py::rand_svd`
(randomized SVD, oversampling=10 default; has a `gesvd`-driver fallback added
2026-07-28 after a reproducible `torch.linalg.svd` non-convergence on
ill-conditioned M_bar during the CA `shared_full` sweep — see §10).
`exactsvd` = full `torch.linalg.svd` + truncate. `countsketch`
(`utils/countsketch.py`) = signed-hash projection of concatenated factors into
`cs_rank` buckets, column-norm-rebalanced first — does not form an optimal
low-rank projection, a JL-style alternative to SVD. `naive_sum` = literal running
sum of raw factors, no SVD, does not preserve delta_W (deliberate — isolates
whether the projection itself matters; requires `svd_rank==lora_rank`).
`nocompress` = keep every singular value above a numerical-rank floor (grows
unboundedly; isolates the cost of bounding rank at all).

**`lazy_merge`** (`off` / `period` / `plateau`, legacy bool → `legacy_saturation`):
let the residual accumulate across multiple cycles instead of folding every
cycle. `period` aliases `svd_period` (same knob, not duplicated — setting both to
different nonzero values raises). `plateau` (`utils/lazy.py::PlateauTracker`,
newest): folds once residual drift `d_c = ||R_c-R_{c-1}||_F/(||R_{c-1}||_F+eps)`
stays below `lazy_merge_delta` for 2 consecutive cycles, or
`lazy_merge_max_holdoff` cycles have elapsed (staleness cap, default 10) —
entirely boundary-blind (no data volume/cycle-count/real-task info read).

**`fd_shrinkage`** (bool): Frequent-Directions "pay rent" — shrinks every kept
squared singular value by the energy of the first discarded one
(`sigma_shrunk² = clamp(Sigma[:l]² - Sigma[l]², min=0)`), bounding sketch growth
instead of monotonic increase. Scoped to `randsvd`/`exactsvd` only. **Decision
(impl_plan_7.28.2026 §0): FD shrinkage is OFF the current path** — cost 2-4 top1
at 15-task Omni locally, no measured accuracy benefit; retained as a
rank/accuracy compression dial (documented off-path, not refuted), long-horizon
(>15-task) claim untested.

## 7. Classifier alignment (CA) — full lineage, config surface, and results

`classifier_alignment` (bool) + `utils/ca.py`. SLCA-style, exemplar-free: online
per-class feature statistics (`ClassStats`, Welford updates) accumulated during
training; head-only AdamW realignment on pseudo-features sampled from those
per-class distributions, run after every fold (or every cycle if lazy).

**v1** (impl_plan_7.27.2026): per-class diagonal `{mean,var}`, 300 steps, batch
128, lr 1e-3 (`ca_cov_mode="diag"`, all v2 kwargs at their off-defaults). Result
at 100MB/15T Omni: **+4.57 top1, −3.67 top5** vs no-CA — real gain, real cost,
motivating the v2 repair sweep.

**v2 variants** (impl_plan_7.28.2026 §2), all at Omni-1K, `bm_budget_mb=100`,
`stop_after_tasks=15`, seed 1993:

| Variant | Key config | top1 | top5 |
|---|---|---:|---:|
| off (no CA) | — | 66.42 | 93.01 |
| v1 / control | `ca_cov_mode=diag`, 300 steps | 70.99 | 89.34 |
| **v2(a) steps100+earlystop** ⭐ | `ca_steps=100, ca_early_stop_patience=<set>` | **71.39** | **89.82** |
| v2(d) logit-adjust-only | `ca_logit_adjust_only=True` (no head retraining, additive per-class prior bias from stored counts) | 66.39 | 90.82 |
| v2(b) shared_full cov | `ca_cov_mode=shared_full` (one pooled 768×768 covariance) | 66.42 | 85.34 |
| v2(b) low_rank_diag cov | `ca_cov_mode=low_rank_diag` (rank-8 + diag, fit from 64-sample reservoir) | 62.62 | 84.71 |
| v2(c) real-mix 50% | `ca_real_mix_frac=0.5` | 64.47 | 82.79 |

**Implied winner: `ca_v2_steps100_earlystop`** — the only variant beating v1 on
BOTH top1 and top5 simultaneously (plan's own selection rule: judge top1+top5
jointly, reject top5 < baseline-1.0). **This verdict is inferred from raw logs,
not recorded in any committed doc.** The natural next step — combining the
steps100/early-stop winner with a covariance-mode winner (arm "e") — is
explicitly sequenced in the plan to run AFTER this sweep and has NOT been run;
since neither covariance variant (shared_full, low_rank_diag) beat diag here,
there may be no covariance-side improvement to combine with, but this is unverified.

**No CA run has ever been executed on any dataset other than OmniBenchmark-1K at
this exact 100MB/15-task setting.** The ImageNet-R grid (§8) uses base
SketchLoRA only, no CA/FD/lazy-merge. Do not conflate the two when reporting.

**Known bug (reproduced, fixed)**: `ca_cov_mode=shared_full`'s pseudo-feature
sampler (`sample_pseudo_features`) crashed 3 times with NaN cross-entropy before
succeeding on the 4th attempt (`_sharedfull_retry3`). Root cause: Cholesky-factor
jitter was a fixed absolute magnitude (1e-5), negligible relative to the pooled
covariance's own scale in some directions → huge pseudo-features → NaN loss →
NaN gradient into `fc` → **because ordinary training's loss backprops through
`fc` into the backbone/adapter every subsequent cycle, the NaN silently poisoned
the adapter's own gradients next cycle too**, manifesting as an unrelated
`torch.linalg.svd` non-convergence crash inside the sketch-fold path one cycle
later. Fixed by scaling jitter to the covariance's own diagonal mean
(`utils/ca.py`) plus the `gesvd` fallback in `rand_svd` (§6). `align_head` also
has a defensive non-finite-loss skip (never let a NaN step reach
`optimizer.step()`).

## 8. Competitor comparisons — actual numbers on record

**Omni-1K, 15 tasks, 50MB, seed 1993** (`exps/round2_anchor/`, the round-2
masked-CE-fixed harness — the number every bolt-on/CA arm above is measured
against; NOTE: this is 50MB, not the 100MB the CA sweep uses — no valid
same-budget other-method comparison exists yet, see below):

| Method | top1 | top5 |
|---|---:|---:|
| InfLoRA | 67.87 | 92.24 |
| O-LoRA | 65.78 | 92.44 |
| TreeLoRA | 64.71 | 93.18 |
| SketchLoRA (frozen, "off") | 64.47 | 92.44 |
| SketchLoRA (legacy lazy-merge) | 53.92 | 87.16 |
| SeqLoRA | 50.25 | 86.55 |

**No SeqLoRA/O-LoRA/InfLoRA/TreeLoRA config exists at `bm_budget_mb=100,
stop_after_tasks=15`** — only SketchLoRA has ever been run at that exact
setting. `exps/planc_step2/*_50mb_15t.json` exist and look like a matching
comparison but are Round-1 harness runs (pre-masked-CE fix) — **retired to
diagnostic-only, never cite these as production numbers**
(`docs/plan_c_bounded_memory_round2.md`). `exps/omni1k_streamcompare/` is the
OLD `stream_run`/`boundary_mode="sample"` design, not `bounded_memory` — not
comparable either.

**ImageNet-R, full 20-task split, bounded_memory, seeds {1993,1996}**
(`exps/imagenetr_grid/`, generated by `scripts/gen_imagenetr_grid_configs.py`,
LOCAL A100 cluster, base SketchLoRA only, no bolt-ons). **Status: KILLED at
20/30 (67%) complete, 2026-07-28 ~18:19 NZST, by explicit user instruction**
("You can kill the remaining jobs from the ImageNet-R runs to get started on
this" — freeing GPUs for the SketchLoRA v2 plan, §5/§7 above). Both 50MB and
100MB budgets are fully complete (both seeds, all 5 methods — the table
below). At 200MB: seqlora/olora/treelora completed for seed 1993 only;
`inflora_200mb_s1993` was killed mid-run (no result, do not treat as a
crash — it was an intentional kill); `sketchlora_200mb_s1993` and the ENTIRE
200mb/seed-1996 group (all 5 methods) were never started. **If resuming this
grid, do not re-run the 20 completed cells** — orchestration lives in
`scripts/imagenetr_grid_queue_v3.sh` (v1/v2 superseded, see script comments
for why; v3 hard-excludes GPU0 from its pool because it was needed for
concurrent SketchLoRA bolt-on runs at the time — re-check GPU availability
before reusing that constraint verbatim). A separate, NEWER, H200-targeted
version of this same grid (`exps/imagenetr_slurm_grid/`, 45 configs = 5
methods x 3 budgets x 3 seeds incl. 1999, `scripts/imagenetr_slurm_grid.slurm`)
was generated 2026-07-31 — see §11.5 — status of its actual SLURM submission
is UNVERIFIED from this host (no direct H200 access, no local `sbatch`/`squeue`
binaries; check with the user or H200-side logs).

| Method | 50MB top1 (s1993/s1996) | 100MB top1 (s1993/s1996) |
|---|---|---|
| O-LoRA | 62.51 / 63.56 | 66.37 / 66.39 |
| TreeLoRA | 59.59 / 60.79 | 62.24 / 64.28 |
| InfLoRA (const. retention 0.975/0.975) | 59.49 / 57.36 | 62.68 / 60.72 |
| SketchLoRA | 52.11 / 54.07 | 53.88 / 55.81 |
| SeqLoRA | 44.36 / 42.99 | 46.86 / 47.05 |

**SketchLoRA is materially weaker relative to competitors on ImageNet-R than on
Omni-1K** — ahead of only SeqLoRA, below O-LoRA/TreeLoRA/InfLoRA at both budgets
checked so far. This is the current open competitiveness gap; 200MB cells may
change the picture but are not yet complete.

**H200, ADMISSIBLE, `round2_slurm_grid`: OmniBenchmark-1K, 30 real tasks,
bounded_memory, 3-seed average (1993/1996/1999)** — the H200-cluster
counterpart grid to the local `round2_anchor`/`imagenetr_grid` numbers above,
no bolt-ons (plain `bounded_eviction`). Originally 4 methods x {100,200}MB
(SeqLoRA/O-LoRA/InfLoRA/SketchLoRA); **extended 2026-08-01 via
`scripts/round2_slurm_grid_completion_and_50mb.slurm`** to 5 methods
(TreeLoRA added, fixing the `_bounded_train_epoch` crash described in §5/§10)
x {50,100,200}MB — all **45 cells (5 methods x 3 budgets x 3 seeds) now
complete at the accuracy-log level** (verified directly by grep'ing every
`run_logs/round2_slurm_grid/<method>_<budget>mb_s<seed>.out` for
`[bounded_mem eval]` lines through `volume 1.00` — no exceptions, no
truncated runs). Final-checkpoint (100% of stream) top1/top5, 3-seed
averaged:

| Method | 50MB top1/top5 | 100MB top1/top5 | 200MB top1/top5 |
|---|---|---|---|
| O-LoRA | 40.71 / 70.41 | 43.10 / 72.90 | 44.66 / 74.58 |
| InfLoRA | 39.07 / 69.08 | 40.63 / 70.86 | 41.85 / 72.19 |
| TreeLoRA | 33.74 / 64.54 | 36.37 / 66.84 | 35.72 / 66.98 |
| SketchLoRA | 29.12 / 59.30 | 29.31 / 58.78 | 26.88 / 56.09 |
| SeqLoRA | 14.97 / 36.94 | 16.01 / 37.62 | 17.99 / 40.91 |

Ordering (O-LoRA > InfLoRA > TreeLoRA > SketchLoRA > SeqLoRA) is stable
across all three budgets. TreeLoRA sits clearly between InfLoRA and
SketchLoRA — consistent with its local-cluster ImageNet-R positioning (§8).

**Persistent-memory data is NOW COMPLETE for the full extended grid**
(as of 2026-08-03 — took two rounds of targeted `scp` from the H200 side,
per-file-pattern rather than a blind full-`run_logs/` copy, since a foreign
session had committed the entire local `run_logs/` tree to the cluster,
making it too large to round-trip wholesale; see `run_one()` in
`scripts/round2_slurm_grid*.slurm` for the exact three-location write
pattern every cell uses — `.out`/`.err` under `round2_slurm_grid/`,
`boundedmem_*.json` at top-level `run_logs/`, `metrics_*.json` under
`run_logs/final/<method>/`). All 45 cells' `metrics_*.json` now read
`"status": "done"` with 10/10 `per_task` entries, TreeLoRA included (its
first two sync attempts brought back a stale `"status": "running"`/0-entry
ghost file from the ORIGINAL pre-fix crash attempt — not a location
problem, the source file on H200 itself needed to catch up; a third sync
resolved it). Final-checkpoint persistent MB, 3-seed averaged:

| Method | 50MB persistent MB | 100MB persistent MB | 200MB persistent MB |
|---|---|---|---|
| TreeLoRA | 1026.0 | 515.5 | 260.3 |
| O-LoRA | 685.0 | 344.7 (unbounded growth) | 174.5 |
| InfLoRA | 91.6 | 92.0 (flat from task 1) | 92.0 |
| SketchLoRA | 19.2 | 17.8 | 17.0 |
| SeqLoRA | 4.3 | 4.3 | 4.3 |

**Headline finding: TreeLoRA has BY FAR the largest and fastest-growing
persistent footprint of any method in this grid** — larger than O-LoRA's
already-unbounded bank at every budget, nearly 3x InfLoRA's at 50MB. Both
O-LoRA and TreeLoRA's memory scales with CYCLE count (one new slot/leaf per
cycle, not per real task), which is why the footprint is highest at the
TIGHTEST budget (50MB → smallest chunks → most cycles → most slots/leaves
accumulated over the same 30-task/1000-class stream) and shrinks
monotonically as budget increases — the same mechanism, just more extreme
for TreeLoRA's per-leaf storage than O-LoRA's per-slot storage. This
directly changes SketchLoRA's/InfLoRA's relative positioning in the
memory-vs-accuracy tradeoff story: TreeLoRA's accuracy edge over SketchLoRA
(§ table above) comes at a MUCH steeper memory cost than O-LoRA's does.

Plots (2026-08-03, final): `run_logs/round2_slurm_grid/plots/
{top1,top5}_accuracy{,_50mb,_200mb}.png` and
`persistent_memory{,_50mb,_200mb}.png` — 5 methods, all 3 budgets, EVERY
panel now complete including TreeLoRA and the 50MB memory panel (which did
not exist before this sync). Built from the authoritative structured
`boundedmem_*.json`/`metrics_*.json` (not log-parsing — cross-validated
against an earlier `.out`-log-parsed pass, values matched exactly).
3-seed-averaged, y-axis 0-100 fixed for accuracy panels only (memory panel
auto-scaled, now spanning up to ~1026MB), matplotlib/seaborn per the
standing PNG-not-artifact rule — see reference in the user's persistent
memory system, §13 below.

**CRITICAL: a separate "final_vision" multi-dataset benchmark
(cifar224/food101/sun397/imagenetr x 6 methods incl. cllora/treelora, 3 seeds)
also exists in `run_logs/final_vision/` from an H200 pull the same day — this
is EXPLICITLY DEFUNCT, old-config work per direct user correction** ("those
aren't the runs we're interested in anymore... old defunct configs... What
we're concerned about are the OmniBenchmark-1k runs, like we've been concerned
about for the past week"). It traces to a 9-days-stale 11-method
multi-dataset plan (originally `FINAL_EXPERIMENTS_IMPLEMENTATION_PLAN.md`,
2026-07-16 to 2026-07-19) that is no longer the project's focus. **Do not
reconstruct, analyze, or cite anything from `final_vision`'s cifar224/
food101/sun397/imagenetr pull as a live finding.** (ImageNet-R also failed
100% on H200 in that pull — all 18 method/seed combos crashed identically on
`FileNotFoundError: ./data/imagenet-r/train/`, a data-prep gap on that
cluster, unrelated to any method — noted here only in case the same missing-
directory issue resurfaces for a CURRENT run that does need imagenet-r on
H200.) The ONLY admissible H200 data as of this writing is the
`round2_slurm_grid` table immediately above. If a future session is asked to
"reconstruct the H200 tables/graphs" without qualification, confirm which
grid is meant before assuming it's `final_vision` just because it's the most
recently modified data on disk.

## 9. Pre-registration discipline (binding on any new analysis)

This project pre-registers predictions verbatim, dated, before seeing results
(`docs/plan_c_preregistration.md` §C6, `impl_plan_7.26.2026` §5), with an
explicit falsification clause. Binding rules for any continuation of this work:
under no branch are retired/Round-1 numbers cited as production; no budget cell
is dropped after seeing results; no new setting variant is introduced post hoc
to change an unwelcome ranking — a redesign is admissible only to fix a
demonstrated artifact (e.g., a failed sanity-anchor gate), never in response to
an ordering the plan didn't predict. When the actual Plan-C money experiment
(Omni-1K, 100MB, 50 tasks, all 4 core methods) showed near-total convergence
(all methods within ~1pt, contradicting the pre-registered spread), this was
reported honestly as the falsification case, reframing the headline claim to
Pareto-only (accuracy-resource tradeoff), not an accuracy win. Maintain this
standard: report contradicting results as contradictions.

## 10. Known architectural failure modes (do not re-investigate; use as given)

- **HiDeLoRA**: collapses to nearly chance accuracy under any streaming/bounded-
  memory design. Root cause: its deployed inference SELECTS a single slot
  (`_predict_task_ids`, merge=False) rather than summing all frozen slots the
  way O-LoRA/InfLoRA/TreeLoRA do — briefly-trained fragmented slots (each
  surviving only 1-2 epochs under frequent folding) are useless to a
  selection-based method but harmless to an additive one. Neither
  warm-start-disable nor momentum-disable ablations fixed it (fully-cold was
  WORSE, not better) — confirmed architectural mismatch, not a bug. Excluded
  from all current work by user decision.
- **ProgPrompt**: CIL forward concatenates every slot from 0 up through the
  CURRENT fold index, but each slot was trained at a much shorter concatenation
  length — severe train/eval sequence-length mismatch, confirmed via a direct
  diagnostic (chunk-count vs CIL-accuracy tracking almost exactly). Status open,
  not currently in bounded_memory grids.
- **Eval-routing bugs** (historical, now guarded by asserts per Plan A §A4.3):
  under the old sample-clock streaming design, TIL eval for O-LoRA/InfLoRA
  routed to the wrong adapter slot (task index != slot index once folds
  outnumber tasks) — fixed via `_stream_task_to_chunk` mapping. Any new
  eval-routing code should assert parameter-state-hash identity at every
  checkpoint (`_bounded_param_hash`, already wired into `bounded_memory_mixin.py`).
- **`_stream_new_optimizer()` never called `_optimizer_param_groups()`** (found
  during Round 2): SketchLoRA's `sketchlora_lora_wd=0.0` setting was silently
  inert under every `stream_run` (non-bounded-memory) run to date. Fixed for
  `bounded_memory` specifically via `_bounded_new_optimizer()`; `stream_run`'s
  own optimizer path deliberately left as-is (demoted harness, out of scope).
- **Round-1 bounded_memory harness bug**: unmasked full-width cross-entropy
  (every class absent from the current cycle was a permanent negative every
  step — logit-suppression). Fixed in Round 2 via cycle-derived class masking.
  All Round-1 numbers are diagnostic-only, never cite as production.

## 11. Uncommitted work as of this writing (verify current git status before assuming)

At the time this document was written, `git status` showed modifications to
`models/{bounded_memory_mixin,inflora,olora,sketchlora,stream_mixin,treelora}.py`
and `utils/randsvd.py`, plus untracked `impl_plan_7.27.2026/`,
`impl_plan_7.28.2026/`, `utils/{admission,lazy,ca,fd,ce_formulas,ops_ledger}.py`,
`exps/{imagenetr_grid,sketchlora_boltons}/`, expanded `exps/round2_grid/`, and
numerous `scripts/*.sh` / `scripts/gen_*.py`. This represents the `floor`
admission rule, the full CA v2 sweep, the CE-metric ledger, and the
(now-killed, see §8) local ImageNet-R grid — i.e., everything in §5-8 above.
This repo IS a real git repo (`git remote -v` confirms
`origin git@github.com:Guy-Marlow/SketchLoRA_Vision.git`, branch `main`, up to
date with origin as of this writing) — `git status`/`git log` are reliable and
should be checked fresh rather than assumed from this doc.

### 11.5. Newer disk state found 2026-07-31 (AFTER this doc's own last edit —
this doc itself can go stale; always diff `git status`/file mtimes against
what's written here before trusting any status claim, including this one)

Four new files, all dated 2026-07-31 (one day after this document's own
2026-07-30 timestamp), untracked, evidently prepared for an H200 SLURM
submission of Section 3's "money rerun" positioning work:
- `scripts/gen_imagenetr_slurm_grid_configs.py` + `exps/imagenetr_slurm_grid/`
  (45 configs = 5 methods x {50,100,200}MB x 3 seeds {1993,1996,1999} — the
  H200-targeted continuation of the killed local ImageNet-R grid, now with a
  3rd seed) + `scripts/imagenetr_slurm_grid.slurm`.
- `scripts/gen_sketchlora_ablations_imagenetr20t_configs.py` +
  `exps/sketchlora_ablations_imagenetr20t/` (12 configs = 4 SketchLoRA
  variants — `current`/`exactsvd`/`fixedrank`/`globaleps` — x 3 seeds; notably
  does NOT include the new `floor` admission rule as a named variant here,
  suggesting either this set predates `floor` or floor is being tested
  elsewhere) + `scripts/sketchlora_ablations_imagenetr20t.slurm`.

**No evidence these were actually submitted to H200** was found from this
host (no `run_logs` entries referencing them, no local `sbatch`/`squeue`
binaries to check queue state, no SLURM `.out`/`.err` files anywhere under
these names). Treat as "configs prepared, submission status unconfirmed" —
verify with the user or by checking for H200-side log copies before assuming
these ran, are running, or produced results.

## 12. How to launch a new run

```
cd LAMDA-PILOT
python3 main.py --config <path-to-json>
```
Config JSON = flat dict of args (see `utils/toolkit.py`/`trainer.py` for the
full key set); `device` field takes a list of GPU index strings (use the
literal `"PLACEHOLDER"` convention + `sed` substitution, matching every
`scripts/*_queue.sh`, if writing a new orchestrator). Config generators live in
`scripts/gen_*.py` (one per campaign; each hardcodes its own `BASE` dict + a
`METHOD_CFG`/`ARMS` table — copy the pattern for a new grid rather than editing
JSON by hand). Results land in `run_logs/<campaign>/` (stdout/stderr log) and
`run_logs/final/<method>/metrics_*.json` (structured metrics via
`utils/metrics_logger.py`, wired into every `bounded_memory` run
unconditionally). `bounded_memory_mixin.py::bounded_memory_run`'s resumability
check requires BOTH the accuracy-results JSON and the metrics JSON to indicate
completion before skipping a config on relaunch — a run that completed under
pre-metrics-fix code will not be silently skipped.

## 13. Separate persistent-memory system (outside this repo, cross-session)

Independently of this file, the operating agent (Claude Code) maintains its
OWN cross-session memory store at
`/home/gmar762/.claude/projects/-home-gmar762-research-continuous-learning/memory/`,
indexed by `MEMORY.md` there. That system persists across DIFFERENT
conversations/sessions (this document persists within/across continuations of
THIS specific research thread, checked into the repo); the two are
complementary, not redundant. Notable entries already there as of this
writing, relevant to continuing this exact project: the two-cluster rule
(§1.5 above originates from that memory, confirm it's still current there
too), a standing rule that research plots are always static PNGs via
matplotlib/seaborn (never web Artifacts) with bright pastel colors and
`context="talk"` sizing, a standing rule that CE/runtime/persistent-memory
metrics must be verified as actually wired into a harness (grep for the
logging calls, never assume), and a note that the old rank8/alpha32/scaling-4
vision convention (2026-07-01) is SUPERSEDED by the current rank10/alpha-null/
scaling-1 convention used everywhere in this document. If picking up this
project from a fresh conversation with no context, check that memory index
too, not just this file.

## 14. Current status snapshot / suggested next steps (as of this writing)

Section 1 (admission rule `floor`) and Section 2 (CA v2 repair sweep) of
`impl_plan_7.28.2026` are functionally DONE — code implemented, tests passing,
sweep complete with a clear winner (`ca_v2_steps100_earlystop`, §7). Neither
has been used in a completed PRODUCTION run yet (only local ablation/smoke
runs) — the natural next step for each:
- **`floor`**: needs a real comparison run (local ranking pass at minimum,
  ideally an H200 production cell) against `bounded_eviction` at matched
  settings, sweeping `admission_floor_k` in {1, 5} per the plan's own spec —
  not yet done.
- **CA v2**: the plan's own next step (arm "e", combining the steps100/
  early-stop winner with a covariance-mode winner) is unrunnable as specified
  since no covariance variant beat `diag` — either re-scope arm "e" (e.g. try
  combining early-stop with `real_mix` at a lower fraction than 50%, which
  wasn't sweeped) or accept `steps100_earlystop` alone as `ca_v2` and move to
  Section 3.
- **Section 3 (H200 money rerun)**: SLURM configs partially exist as of
  2026-07-31 (§11.5) but submission is unconfirmed — this needs a human with
  H200 access; report status, don't assume completion.
- **Section 4 (InfLoRA positioning)**: the CE-hook audit (§3 above) found a
  REAL, quantified gap — `inflora_boundary_macs()` only charges the
  incremental covariance-bookkeeping cost of InfLoRA's two extra full passes
  per cycle, never their own base forward-pass compute, understating
  InfLoRA's true overhead by roughly 4x (reported CE ~0.993-0.994 vs a
  back-of-envelope corrected ~0.96). This needs a DECISION (fix the ops-ledger
  accounting project-wide, which changes every existing InfLoRA CE number, vs.
  document as a known caveat) that has NOT yet been made — do not silently
  "fix" this without surfacing the tradeoff first, per this project's own
  pre-registration discipline (§9) against post-hoc changes that alter
  reported numbers.
