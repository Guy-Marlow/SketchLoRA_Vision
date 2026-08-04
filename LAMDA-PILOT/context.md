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
   production runs use this harness** — **EXCEPT** the
   `sketchlora_ablations_imagenetr20t` campaign (§8.7), which deliberately runs
   regime 1 (plain oracle CIL, no `boundary_mode` key at all) on the full
   ImageNet-R 20-task split, specifically to isolate SketchLoRA's own
   merge/rank/CA mechanisms from any bounded-memory-streaming interaction.
   Don't assume every new SketchLoRA config is bounded_memory just because
   that's the general rule — check for the presence/absence of
   `boundary_mode` in the config itself.

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
    models/                           one file per CL method, mixins in lora.py;
                                       noadapt.py (2026-08-05) = no-adaptation
                                       floor baseline, frozen ViT + NCM head
    backbone/                         ViT variants (vit_lora.py = shared LoRA scaffold)
    utils/                            randsvd (+ rand_svd_probe/factors_from_probe,
                                       2026-08-03 fix, see §8.5), countsketch, admission,
                                       lazy, ca, fd, ce_formulas (LEGACY, superseded by
                                       ce_profiler for anything that matters, see §8.6),
                                       ce_profiler (2026-08-03, measured-CE region
                                       profiling), ops_ledger, data_manager, metrics_logger
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
| SketchLoRA | `models/sketchlora.py` | **O(1)**: two slots, sketch+residual | see §4-6; **rand_svd merge_op has a fixed 2026-08-03 accuracy bug, see §8.5 before trusting any pre-that-date number** |
| noadapt | `models/noadapt.py` (2026-08-05) | O(1): frozen backbone, no adapter ever trained | no-adaptation floor baseline — frozen ViT-B/16 (LoRA slot stays zero-init no-op forever), classifier head fit by one closed-form nearest-class-mean pass/task, ZERO gradient steps of any kind. Oracle-regime only (§8.7), not part of the bounded_memory grids. Purpose: how much of any method's accuracy is just the pretrained backbone's already-near-linearly-separable features vs. real continual adaptation. |
| HiDeLoRA | `models/hidelora.py` | — | EXCLUDED from all current work — genuinely broken under any streaming/bounded-memory design (see §10, "HiDeLoRA collapse") |
| ProgPrompt | `models/progprompt.py` | — | status open/unresolved under streaming (sequence-length train/eval mismatch, §10); not in current bounded_memory grids |
| CL-LoRA, TUNA, EASE, RainbowPrompt | native/ported | — | oracle-grid only; not part of the bounded_memory task-agnostic track |
| SeqLoRA (oracle, ImageNet-R 20t) | `models/lora.py` (base Learner), `_train_adapter`/`_eval_adapter` pinned to 0 | O(1) | added to `sketchlora_ablations_imagenetr20t` (§8.7) 2026-08-05 as a comparison point, standard HPs (lr 3e-4, rank 10, α-null) — same convention as the bounded_memory roster's SeqLoRA row above, just run under the oracle regime this once |

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

## 8.5. SketchLoRA `rand_svd` double-SVD accuracy bug (fixed 2026-08-03, commit `886455b`) — READ BEFORE TRUSTING ANY PRE-FIX NUMBER

`_compress()`'s adaptive-rank path (`svd_energy_target` set — the default,
used by `current`/`exactsvd`/`globaleps`/`exactsvd_ca` in every campaign to
date) was calling `torch.linalg.svdvals(delta_W)` — an EXACT SVD of the full
matrix — to decide the target rank `r_hat_t`, then SEPARATELY calling
`rand_svd()` (its own independent randomized decomposition) to build the
actual `(B_hat, A_hat)` factors. The rank decision was using perfect spectral
knowledge a randomized method should never have access to — a real deviation
from the specified algorithm, not just wasted compute. **This means every
SketchLoRA result in this document (and every SketchLoRA number anywhere in
this repo) dated before 2026-08-03 that used `merge_op="randsvd"` +
adaptive rank — i.e. every §8 table above (`round2_anchor`, `round2_slurm_grid`,
`imagenetr_grid`), the whole CA v1/v2 sweep in §7, everything — was produced
under this bug.** Do not treat those numbers as reflecting the algorithm as
specified; they reflect a strictly easier rank-selection procedure than
`randsvd` is supposed to have.

**Fix** (`utils/randsvd.py`: new `rand_svd_probe()` + `factors_from_probe()`;
`models/sketchlora.py::_compress()`): the randomized decomposition now runs
ONCE, sized to `composite_rank = prev_rank + residual_total` (an EXACT upper
bound on `delta_W`'s true rank, derived from the LoRA factor structure, not
estimated), and both the rank decision and the final factors come from that
single decomposition. The same bug pattern still exists, UNFIXED, in
`utils/admission.py::floor_admission_merge` (only reachable via
`sketchlora_admission="floor"`, not used in any production config to date —
lower priority but real, fix before ever using `floor` for a reported number).

**Reassuring finding post-fix** (§8.7's `sketchlora_ablations_imagenetr20t`
campaign, run entirely under the fixed code): `exactsvd`'s accuracy (which
was never affected by this bug — it always used a genuine full exact SVD)
tracks `current`/randsvd's POST-fix accuracy within ~0.5-1 point per seed —
the fix did not leave `randsvd` meaningfully behind the "ideal" exact-SVD
ceiling. This is real evidence the fix didn't break anything, but it is not
the same claim as "the pre-fix numbers above are still valid" — they aren't;
the fix changed what the algorithm actually does, even if the net accuracy
effect turned out to be small on this particular split.

## 8.6. Measured-CE profiling system (2026-08-03, supersedes `ce_formulas.py` for anything reported)

The original CE (computational-efficiency) accounting — `utils/ce_formulas.py`,
hand-derived analytic MAC formulas per method, referenced throughout §3/§8
above — was abandoned after three separate formula bugs were found by manual
re-reading (InfLoRA missing its own base-forward cost, entirely the subject
of §3's "CE-metric audit finding" above; InfLoRA's flat `dim^3` DualGPM term
ignoring basis growth; TreeLoRA's flat per-step formula plus a zero boundary
term). Replaced with direct `torch.profiler` region measurement:
`utils/ce_profiler.py` (`ce_region(label)` context-manager tag, wrapped in a
no-op when no profiling session is active so tags are safe to leave in
permanently; `CEProfileSession`; `CEProfileController`, which handles
sampling cadence + hold-between-cycles logic), `utils/ops_ledger.py`
(`OpsLedger.record_unit()` extended with `measured_step_regions`/
`measured_boundary_regions`/`baseline_step_macs_fwd`/`baseline_step_macs_bwd`/
`profile_provenance`; `compute_ce(source=, baseline_numerator=)`;
`compute_ce_report()` returns `ce_formula`/`ce_measured`/
`ce_formula_baseline_numerator`/`ce_measured_baseline_numerator` plus
`ce_best`/`ce_best_source`, a preference-ordered pick of the most trustworthy
variant actually available — **this is the value to log/read, never
`ce_formula` alone**, which is an inert ~1.0 under oracle mode by
construction since the old aux/boundary formula hooks are never called
there).

**The fairness rule this whole redesign encodes**: "would SeqLoRA also pay
this cost?" If yes, it's shared baseline, never charged as a method's own
overhead. The single biggest fairness hole this fixes: a naive per-method
`Ops_fb` measurement lets a method's own extra forward cost (O-LoRA/
InfLoRA/TreeLoRA's frozen-delta matmul, SketchLoRA's sketch-inclusion matmul)
sit in BOTH the CE numerator and denominator and cancel, making an
expensive-forward method with no aux cost read CE=1.0 and look free. Fix
("R2" in the implementation notes): measure a SeqLoRA-equivalent baseline
(single slot, `merge=False`) via `measure_baseline_and_actual()` and use it
as a shared numerator correction — this is what `ce_best_source` resolving
to `ce_measured_baseline_numerator` means in every log line you'll see.

**Two structurally different wiring paths, gated behind `final_metrics=true`
(same flag `MetricsLogger` uses)**:
- `models/bounded_memory_mixin.py` (the streaming driver): THREE profiling
  kinds per cycle — `boundary_begin` (`_stream_begin_chunk`), `step` (epoch 0
  of a sampled cycle only), `boundary_end` (`_stream_end_chunk`) — because the
  driver calls these as three separate, driver-visible steps and can wrap
  each independently.
- `trainer.py`'s oracle per-task loop (added 2026-08-03 specifically for this
  — the oracle path had ZERO CE logging of any kind before): only ONE
  controller kind, `"task"`, wrapping the entire opaque
  `model.incremental_train(data_manager)` call (each method decides
  internally when to run its own boundary actions; trainer.py has no hook
  into that internal structure). Routed into `measured_boundary_regions`
  (a one-off per-task cost), NOT `measured_step_regions` (which gets
  multiplied by `n_epochs*steps_per_epoch` downstream and would wildly
  overcount a whole-task profile). Coarser than bounded_memory's 3-way split
  — no separate "per-step-only" vs "boundary-only" view under oracle mode —
  but not double-counted or mis-scaled.

**`ce_profile_every` semantics — the single most important gotcha for
reading any CE number in this repo**: this controls ONLY the expensive
full-region `torch.profiler` trace (`measured_step_regions`/
`measured_boundary_regions`, what `n_actually_profiled` in a `[CE metric]`
log line counts). It does NOT gate the cheap, always-on single-batch R2
baseline probe (`measure_baseline_and_actual`, called unconditionally
whenever `final_metrics` is on) — so `ce_profile_every=0` (used throughout
`sketchlora_ablations_imagenetr20t`, §8.7, because profiling a WHOLE
`incremental_train()` call — every epoch, every step, real data — over 20
real tasks is expensive) still produces a real, non-1.0 `ce_best` value via
`ce_measured_baseline_numerator`; it just means `n_actually_profiled: 0` and
no per-region breakdown exists for those runs. `ce_profile_every=1` (used in
the staged-but-not-yet-submitted `ce_smoke_imagenetr5t` 5-task validation
run, `scripts/ce_smoke_imagenetr5t.slurm` + its config generator) profiles
every task and IS meant to give the full region breakdown, at real cost (took
~40min for 4 tasks of the cheapest method in early testing) — appropriate
only for a short validation run, not a full campaign.

**Known unresolved, flagged not silently assumed**: whether
`torch.profiler.record_function` correctly attributes a *backward* pass back
to a forward-tagged region is unverified against real torch 2.4.1 behavior
(O-LoRA's `orth_penalty_matmul` tag only gives a measured FORWARD floor).
O-LoRA's `orth_prev_cache_rebuild` fires once per CYCLE not once per EPOCH
(unlike TreeLoRA's analogous cache) — deliberately left unmodeled rather than
force-fitting the standard per-step scaling pipeline, which would overstate
it ~20x. `utils/admission.py::floor_admission_merge`'s same exact-SVD-for-
rank-selection pattern as §8.5's bug is unfixed there too (ablation-only,
lower priority). Full implementation detail: `docs/ce_profiling_implementation_plan.md`.

**Nothing in this CE-profiler subsystem has been GPU-tested as of this
writing** — every line was verified only by `ast.parse`, real Python imports,
and synthetic (fabricated-input, no `torch.profiler`) unit tests of the
ledger/controller logic, per the two-cluster rule (§1.5) — local GPUs are
off-limits for anything beyond that. The `ce_smoke_imagenetr5t` 5-task
ImageNet-R oracle smoke test exists specifically to validate this on a real
H200 run before trusting any number it produces, and — as far as this
document's own authors know — has not yet been confirmed submitted.

## 8.7. `sketchlora_ablations_imagenetr20t` campaign (2026-08-04/05) — real H200 results, a real bug found and fixed, a repair round staged

6 SketchLoRA merge/rank/admission variants x 3 seeds (1993/1996/1999),
ImageNet-R FULL 20-task split (`init_cls=10, increment=10`), ORACLE
boundaries (no `boundary_mode` key — the §1 exception), `final_metrics=true`,
**`ce_profile_every=0`** (region-level CE breakdown deliberately off, see
§8.6 — only the cheap formula+baseline-probe CE path populates every
record). Full per-variant reasoning: `scripts/gen_sketchlora_ablations_imagenetr20t_configs.py`'s
own docstring (long, careful, read directly rather than re-derived here).

**The 6 original variants**, each changing exactly one axis off `current`
(the frozen-v1 production config used everywhere else in this repo):
`current` (adaptive rank ε=0.01, `bounded_eviction`, `rank_cap=128`,
`merge_op=randsvd`), `exactsvd` (`merge_op=exactsvd`), `globaleps`
(`sketchlora_admission=global_eps`, removes the bounded_eviction per-merge
cap), `fixedrank` (`svd_energy_target` unset → rank pinned at
`svd_rank=lora_rank=10`, never adapts; admission forced to `global_eps`),
`exactsvd_ca` (`exactsvd` + `classifier_alignment=True`, `ca_steps=300`/
`ca_batch=128`/`ca_lr=0.001`), `countsketch` (`fixedrank`'s sibling:
`merge_op=countsketch`, fixed rank for the same "no spectrum to threshold on"
reason).

**Run status (SLURM job 22181, one 12h allocation, started 2026-08-04
03:42 NZST)**: 16/18 cells completed fully (all 20 tasks).
`countsketch_s1996` was cut mid-run at task 10/20 when the wall clock hit (no
partial-epoch checkpointing — a resubmit reruns it from task 0, not
resumes); `countsketch_s1999` never started. `scripts/sketchlora_ablations_imagenetr20t.slurm`
is resumable (skips any cell whose `metrics_*.json` already shows
`"status": "done"`, lockfile-guarded) — resubmitting it picks up just those
2 remaining cells, does not need to be folded into anything else.

**Local-copy caveat**: only the per-cell `.out` stdout files + the SLURM
job's own `.out`/`.err` were ever copied back to
`run_logs/sketchlora_ablations_imagenetr20t/` — the actual
`metrics_*.json`/`ops_ledger_*.json` (which live under
`run_logs/final/sketchlora/` on the H200 side) were never transferred. Every
number below was extracted by parsing the captured stdout
(`Average Accuracy (CNN)` / `CNN top5 curve` / `[CE metric] sketchlora = ...`
lines), not from the structured JSON.

**Results (final avg CIL top1/top5 over all 20 tasks, mean of complete
seeds; CE = `ce_best`, source `ce_measured_baseline_numerator` for every
record since `ce_profile_every=0`)**:

| variant | top1 mean | top5 mean | CE |
|---|---:|---:|---|
| current | 62.67 | 83.68 | ~0.982 |
| exactsvd | 63.22 | 83.80 | ~0.982 |
| exactsvd_ca | *(bug — see below, byte-identical to `exactsvd` as originally run)* | | |
| globaleps | 62.56 | 83.46 | ~0.983 |
| fixedrank | 59.05 | 80.45 | ~0.996 |
| countsketch | **17.32** (s1993 only, complete) | 26.86 | ~0.996 |

**Key findings:**
1. **CountSketch merge collapses catastrophically** — 17.3% avg CIL top1 vs
   59-64% for every SVD-based variant, a ~42-point gap. The partial `s1996`
   run (already at 33.6% by task 10/20, per-task curve falling into single
   digits by task 6-9) is consistent. `utils/countsketch.py` was audited
   directly before this campaign and looks sound (handles the all-zero-norm
   edge case, deterministic per-(task,module) seed) — working hypothesis is
   this is a real algorithmic result (hash-based rank reduction genuinely
   destroys task-discriminative structure SVD truncation preserves), not an
   implementation bug. Needs `countsketch_s1996`/`s1999` to finish before
   citing 17.3 as more than a single data point.
2. **`fixedrank` costs ~3.6 points top1 (~3.2 top5) vs `current`'s adaptive
   threshold** — real, consistent across all 3 seeds. Adaptive rank selection
   is earning its complexity.
3. **`globaleps` is statistically indistinguishable from `current`** (62.56
   vs 62.67 mean) — on this split/budget, disabling the `bounded_eviction`
   eviction cap doesn't move accuracy, contrary to what its existence (as a
   fix for "rank collapse") might suggest. Worth revisiting on a longer
   horizon or tighter budget with more merge events before concluding the
   cap is unnecessary in general.
4. **`exactsvd_ca` was a silent no-op — `classifier_alignment` never
   actually ran, root cause confirmed not speculated.** `exactsvd_ca`'s CNN
   top1/top5 curves were BYTE-IDENTICAL to plain `exactsvd`'s on every seed.
   Traced: the actual `align_head`/`apply_logit_adjustment` call lived
   entirely inside `models/sketchlora.py::_stream_end_chunk`, a
   `StreamMixin` hook only ever invoked by `bounded_memory_mixin.py`'s or
   `stream_mixin.py`'s own drivers — NEVER by `trainer.py`'s plain oracle
   `incremental_train()` path, which is what this entire campaign uses (the
   §1 exception). SketchLoRA still compressed correctly under oracle mode
   because it separately overrides `_train()` to fold at period boundaries —
   but that override never touched CA, so `self._ca_stats` was never even
   constructed and the whole CA branch was dead code: no crash, no log line,
   silently skipped.

**Fixed 2026-08-05, commit `543a4fc`**, in `models/sketchlora.py`: new
shared helpers `_ca_lazy_init_stats()` (idempotent lazy build of
`self._ca_stats`) and `_ca_reset_reservoir()` (extracted unchanged from
`_stream_begin_chunk`'s existing per-cycle reset) are now called from BOTH
`_stream_init`/`_stream_begin_chunk` (streaming, unchanged) AND the oracle
`_train()` override (new). New `_run_ca_alignment()` — the `align_head`/
`apply_logit_adjustment` dispatch, mechanically extracted from
`_stream_end_chunk` bit-exact (same `ce_region` tags) — is called from both
`_stream_end_chunk` (unchanged) and the end of oracle `_train()` (new),
matching "every cycle" cadence (a task IS a cycle under oracle mode with
`svd_period=1`, the value every production/ablation config uses). New
`_train_with_ca()` is the oracle-path counterpart to the streaming path's
`_bounded_train_epoch`: reimplements `models/lora.py::_train`'s multi-epoch
loop so the same forward pass that computes the training loss also feeds
`ClassStats.update`/the real-feature reservoir — no extra forward pass,
mirroring `_bounded_train_epoch`'s own design. Verified with a synthetic
CPU-only unit test (fake network standing in for the real ViT, per §1.5 — no
local GPU execution): `ClassStats` builds and populates correctly,
`align_head`/`apply_logit_adjustment` demonstrably change `net.fc`'s
weights, lazy-init is idempotent across a simulated next task, the reservoir
resets correctly. **Still not GPU-tested for real** — the repair campaign
below is what validates it.

**`models/noadapt.py` added 2026-08-05** (registered in `utils/factory.py`
as `model_name="noadapt"`) — see the roster table in §3.

**Repair/extension campaign, staged 2026-08-05, NOT yet run** (no direct
H200 access from this session — §1.5). Two independent SLURM scripts, same
resources each (1 GPU, 10 CPU cores, 48GB RAM, 3h30m wall — split into two
scripts and given this exact wall time at explicit user request):
- `scripts/sketchlora_ablations_imagenetr20t_repair_a.slurm` (9 cells):
  `exactsvd_ca` FORCE-rerun once (bypasses the pre-existing `"status": done`
  marker from job 22181's corrupted run — the one cell in either script that
  does this — overwrites `run_logs/final/sketchlora/{metrics,ops_ledger}_..._exactsvd_ca_..._s<seed>.json`
  in place; its own stdout capture is `exactsvd_ca_s<seed>.out`, deliberately
  a different filename from job 22181's `sketchlora_exactsvd_ca_s<seed>.out`
  so the old corrupted-run stdout survives alongside it as a historical
  record) + `seqlora` (new comparison point, see §3) + `noadapt` (new
  baseline, normal skip-if-done for both).
- `scripts/sketchlora_ablations_imagenetr20t_repair_b.slurm` (6 cells, both
  brand new, normal skip-if-done): `fixedrank_ca` (= `fixedrank` + CA, same
  CA hyperparameters as `exactsvd_ca` — isolates how much of `fixedrank`'s
  ~3.6pt deficit vs `current` is closable by correcting classifier HEAD
  drift alone vs. requiring the adaptive rank itself) + `fixedrank_exactsvd_ca`
  (`fixedrank_ca` but `merge_op=exactsvd` — how much further exact-SVD
  reconstruction recovers once CA has already been credited).
  Config generator for all 4 new configs (shared by both scripts, idempotent
  if run twice): `scripts/gen_sketchlora_ablations_imagenetr20t_repair_configs.py`.
  Independent scripts (share only the generator + output directory) — safe
  to submit both at once. Neither touches `current`/`exactsvd`/`globaleps`/
  `fixedrank`/`countsketch`'s own outputs (different tags, never opened).
  All 15 new `.out` filenames (`<variant>_s<seed>.out`, no `sketchlora_`
  prefix) were checked against every existing filename in
  `run_logs/sketchlora_ablations_imagenetr20t/` — no collisions.
  `countsketch_s1996`/`s1999` (job 22181's own incomplete cells) are
  explicitly OUT OF SCOPE for both repair scripts — resubmit the ORIGINAL
  `scripts/sketchlora_ablations_imagenetr20t.slurm` separately for those.

**Known bug in the diagnostics output, found 2026-08-05, NOT fixed** — real,
pre-existing (not introduced by the repair scripts), but directly bites
them: `models/sketchlora.py`'s `_diag_path` (`sketchlora.py:343`, the path
`run_logs/sketchlora_diag_*.json` that `sketch_diag=True`'s per-compression
retained-energy/r̂ records get written to) is keyed only by
`(energy_target-or-svd_rank, n_lora_blocks, split, seed)` — NOT by
variant/prefix/`merge_op`. Two collision groups exist across the whole
campaign: `adapt0.01_ball_ic10i10_seed<seed>` (shared by `current`,
`exactsvd`, `globaleps`, `exactsvd_ca`) and `r10_ball_ic10i10_seed<seed>`
(shared by `fixedrank`, `countsketch`, `fixedrank_ca`,
`fixedrank_exactsvd_ca`). Each variant sharing a tag opens the same JSON
path and overwrites it (`"w"` mode) on every compression event, so only the
LAST variant to run in that group (by execution order) has a surviving file
on disk — this already silently happened during job 22181 (`current`'s and
`exactsvd`'s own diag files no longer exist; `exactsvd_ca` clobbered them
last) and WILL recur in `repair_b` (`fixedrank_ca` then
`fixedrank_exactsvd_ca` run sequentially in the same script, same tag —
`fixedrank_ca`'s diag JSON will be lost). Does NOT affect accuracy/CE
numbers (those live in per-run `metrics_*.json`/stdout, unaffected) and does
NOT affect the `[SketchDiag]` stdout log lines (each variant's own `.out`
file keeps its own lines regardless — only the consolidated JSON is lossy).
**Not fixed** — would require making `_diag_path` include `args.get("prefix")`.
Do this before submitting either repair script if per-variant compression
diagnostics (not just stdout) matter for this round.

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

## 11. Git state as of this writing (2026-08-05, commit `543a4fc`) — verify current git status before assuming

As of this writing, `origin/main` is at `543a4fc` ("Fix SketchLoRA
classifier_alignment dead under oracle-mode training; add no-adaptation
baseline; stage repair/extension campaign") and the working tree is clean
with respect to every file this document describes — §8.5-§8.7's fix,
`models/noadapt.py`, both repair `.slurm` scripts, their config generator,
and the 12 new `exps/sketchlora_ablations_imagenetr20t/*.json` configs are
ALL committed and pushed. Commit history for this whole arc, newest first:
`543a4fc` (§8.7's fix + repair campaign) → `d9326b0` (resource tuning on the
original ablations script) → `45db144` (original `sketchlora_ablations_imagenetr20t`
campaign, 6 variants x 3 seeds) → `14257dd` (ImageNet-R data-path symlink
fix, `./data/imagenet-r` → `./data/imagenetr`, backported into both
`ce_smoke_imagenetr5t.slurm` and `imagenetr_slurm_grid.slurm`) → `886455b`
(§8.5/§8.6's fix — measured-CE profiling + the rand_svd bug fix).

**What IS still uncommitted / untracked, and should stay that way** (an
inherited, pre-existing pattern from other work in this repo, not part of
anything §8.5-§8.7 touched — do not stage or commit these without a specific
reason): extensive modifications/deletions under `run_logs/final_vision/`
and `run_logs/round2_slurm_grid/` (stale plots/logs churn from an unrelated,
defunct campaign, §8's "final_vision" callout), plus a handful of untracked
files (`exps/imagenetr_slurm_grid/`, `exps/round2_slurm_grid/_smoke_cefix_*.json`,
`scripts/gen_imagenetr_slurm_grid_configs.py`) whose origin/purpose was never
fully investigated by the sessions that did the §8.5-§8.7 work — they were
deliberately left alone (not staged, not deleted) rather than guessed about.
If a future session needs to touch `run_logs/` or these specific untracked
files, investigate their origin first rather than assuming they're safe to
discard or commit.

### 11.5. `imagenetr_slurm_grid` — status unconfirmed, separate from §8.7

`scripts/gen_imagenetr_slurm_grid_configs.py` + `exps/imagenetr_slurm_grid/`
(45 configs = 5 methods x {50,100,200}MB x 3 seeds {1993,1996,1999}, the
H200-targeted continuation of the (killed, §8) local ImageNet-R
`bounded_memory` grid) + `scripts/imagenetr_slurm_grid.slurm` — this is a
DIFFERENT campaign from `sketchlora_ablations_imagenetr20t` (§8.7): it's the
5-method `bounded_memory` budget grid (regime 3), not the oracle-regime
SketchLoRA-only ablation series (regime 1). **No evidence this grid was
actually submitted to/completed on H200** has ever been confirmed from any
session with access to this repo (no local `sbatch`/`squeue` binaries to
check queue state). Treat as "configs prepared, submission status
unconfirmed" — verify with the user or by checking for H200-side log copies
before assuming this ran, is running, or produced results. (Its own data
symlink fix, `14257dd`, IS committed regardless of whether the grid itself
was ever run — see §11's commit list.)

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

## 13. Separate persistent-memory system (outside this repo, NOT git-portable — read this if you're a fresh session on a clone)

Independently of this file, the operating agent (Claude Code) maintains its
OWN cross-session memory store at
`/home/gmar762/.claude/projects/-home-gmar762-research-continuous-learning/memory/`,
indexed by `MEMORY.md` there. **That system is local to one specific
machine's Claude Code install and is NEVER committed to this git repo — it
does not exist in a fresh `git clone`.** This document (`context.md`) is the
ONLY mechanism in this project that actually travels with the repo — that is
its entire purpose, and it is why it is kept this dense: **if you are a
Claude instance reading this after a fresh clone on a different machine, this
file plus code comments (this codebase leans heavily on long, careful header
comments in `.slurm`/config-generator scripts specifically so a fresh reader
doesn't need the memory system to reconstruct intent — see any
`scripts/*.slurm` file for examples) are ALL the context you have. Do not
assume access to anything under `~/.claude/projects/`, do not reference it,
and do not ask the user to "check memory" — it isn't there for you.**

Notable content that lived in that memory system as of 2026-08-05, already
folded into this document above so a clone doesn't need it: the two-cluster
rule (§1.5), the CE-instrumentation redesign and fairness conventions (§8.6),
the rand_svd bug (§8.5), the `sketchlora_ablations_imagenetr20t` campaign and
its CA bug fix (§8.7), the rank10/alpha-null/scaling-1 convention superseding
the old rank8/alpha32/scaling-4 one (§3). Standing STYLE/PROCESS rules that
are NOT project facts and so are not repeated in this doc, but worth knowing
if you're continuing this work under the SAME operator who has stated them
before: research plots are always static PNGs via matplotlib/seaborn (never
web Artifacts), bright pastel colors, `context="talk"` sizing; before
proposing a speed optimization to a method WITH a published reference
implementation, trace what the reference's own code actually does first —
don't assume a naive port is "our inefficiency" to fix without checking (the
O-LoRA orthogonality-penalty vectorization, §3, was reframed this way); for
code with NO reference (SketchLoRA's own `rand_svd`/CountSketch), the bar for
any optimization is bit-identical output (`torch.equal`), not merely
`allclose` — a batched-GPU-QR/SVD optimization was tested and declined on
exactly this bar (~1e-4 drift, not bit-exact). If continuing this project
in a fresh session where that memory system IS available (i.e., same machine,
not a clone), it remains useful for finer-grained blow-by-blow history than
this document carries — but treat this document as authoritative for
anything the two disagree on, since memory entries are frequently stale
point-in-time snapshots and this file is the one artifact both sides of this
project (human and any Claude instance, on any machine) are expected to keep
in sync.

## 14. Current status snapshot / suggested next steps (as of 2026-08-05)

**Most immediate, most likely to be what a fresh session is picked up to do:**
1. **Submit both `scripts/sketchlora_ablations_imagenetr20t_repair_a.slurm`
   and `_repair_b.slurm` on H200** (§8.7) — staged, committed, pushed, not
   yet run; no session working from this repo alone has H200 access (§1.5),
   this needs the human operator. Before submitting, decide whether to fix
   the `_diag_path` collision bug (§8.7's last paragraph) — cheap to fix,
   currently silently drops `fixedrank_ca`'s own compression-diagnostics JSON.
2. Once repair_a/b land: confirm `exactsvd_ca` now differs from `exactsvd`
   (the whole point of the CA/oracle-mode fix — if the curves are STILL
   identical, the fix itself has a bug, don't assume success); read
   `seqlora`/`noadapt` as new floor/reference points; use `fixedrank_ca` vs
   `fixedrank` vs `current` to decompose how much of the adaptive-rank
   advantage is classifier-head drift vs. rank itself.
3. Resubmit the ORIGINAL `scripts/sketchlora_ablations_imagenetr20t.slurm`
   separately to finish `countsketch_s1996`/`s1999` (out of scope for the
   repair scripts, §8.7) — needed for a stable 3-seed countsketch-collapse
   result before citing it anywhere.
4. Copy back `run_logs/final/{sketchlora,seqlora,noadapt}/*.json` and
   `run_logs/sketchlora_diag_*.json` from H200 once runs complete — neither
   repair script does this automatically, and this document's own §8.7
   numbers are stdout-derived only, not sourced from the structured JSON.

**Older, lower-priority open items, still genuinely open as of this
writing** (carried forward from the pre-2026-08 state of this document; no
session since has picked these up, they are not implicitly superseded by
anything in §8.5-§8.7):
- **Admission rule `floor`** (§5): implemented, unit-tested, never used in a
  completed production run. Needs a real comparison against
  `bounded_eviction` at matched settings, sweeping `admission_floor_k` in
  {1, 5}. Also still carries §8.5's rand_svd-adjacent exact-SVD-for-rank-
  selection bug in `floor_admission_merge` specifically — fix that first if
  `floor` is ever used for a reported number.
- **CA v2 arm "e"** (§7): unrunnable as originally specified (no covariance
  variant beat `diag`) — needs re-scoping (e.g. combine early-stop with a
  lower `real_mix_frac`) or explicit acceptance of `steps100_earlystop` alone
  as `ca_v2`.
- **`imagenetr_slurm_grid`** (§11.5): submission status still unconfirmed.
- **InfLoRA CE-accounting gap** (§3's "CE-metric audit finding"): a real,
  quantified ~4x understatement of InfLoRA's true overhead in the OLD
  `ce_formulas.py` accounting. §8.6's measured-CE system may have already
  superseded this concern for any NEW InfLoRA run (it measures rather than
  estimates), but the historical InfLoRA CE numbers cited in §3/§8 above were
  never revisited under the new system — if InfLoRA's CE is reported again,
  confirm it's coming from `compute_ce_report`'s `ce_best`, not the old
  `ce_formulas.py` path, before trusting it.
