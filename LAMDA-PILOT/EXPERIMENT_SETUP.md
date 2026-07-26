# Experimental Setup

Living document describing the datasets, task splits, and per-method
hyperparameters used in the final vision continual-learning evaluation.
Built up incrementally; sections are added as they're settled.

## Backbone Consistency (2026-07-21)

All 10 methods must use the same pretrained ViT-B/16 backbone. Traced every
method's actual `timm.create_model(...)` call and found two outliers: TUNA
and CL-LoRA were both pointed at the pure-IN21K checkpoint
(`vit_base_patch16_224_in21k_*`), while every other method (SeqLoRA/
SketchLoRA/O-LoRA/InfLoRA/TreeLoRA/RainbowPrompt/EASE/ProgPrompt) — and, per
a full survey of every `exps/*.json` LAMDA-PILOT bundles, essentially the
*entire* rest of the toolkit's ~25+ method catalog including ones we haven't
implemented — uses the no-suffix, IN1K-finetuned checkpoint
(`utils/inc_net.py:13-17`: both are IN21K-pretrained; the no-suffix variant
adds an IN1K fine-tuning stage on top). TUNA/CL-LoRA were the outliers, not
the other 8 — both were ported from their own separate reference repos rather
than being native LAMDA-PILOT baselines, likely explaining the deviation.
Fixed by switching both to their existing in1k-variant backbone functions
(`vit_base_patch16_224_tuna`, `vit_base_patch16_224_cllora`) across every
config file in the tree (`backbone_type` field only — no code changes).

## Dataset Splits

All splits verified exactly against each dataset's true class count via
`utils.data_manager.DataManager` (`sum(get_task_size(t) for t in range(nb_tasks))
== nb_classes`, and `nb_tasks` matches the target task count) — no remainder or
off-by-one gaps. `init_cls` absorbs whatever remainder is left after the
other `n_tasks - 1` tasks of a clean `increment`, when the total doesn't
divide evenly.

| dataset | classes | init_cls | increment | tasks |
|---|---|---|---|---|
| ImageNet-R | 200 | 10 | 10 | 20 |
| CIFAR-100 | 100 | 10 | 10 | 10 |
| SUN397 | 397 | 37 | 40 | 10 |
| Food101 | 101 | 6 | 5 | 20 |
| OmniBenchmark-1K | 1000 | 10 | 10 | 100 |

Notes:
- SUN397 (397) is prime, so no split of 10 tasks can be perfectly even; 37/40
  (task 0 slightly smaller, the other 9 equal) was chosen over the alternative
  10/43 to keep the non-uniform task closer in size to the rest.
- CIFAR-100, Food101 (101 = 6 + 19×5), and OmniBenchmark-1K (1000 = 100×10,
  perfectly even) all divide cleanly or near-cleanly.
- OmniBenchmark-1K is back in scope as of 2026-07-20 (previously dropped
  2026-07-19 for being under-studied/longest-running); ImageNet-R's split is
  unchanged from the prior convention. CIFAR-100/SUN397 move from 20-task to
  10-task splits, replacing the earlier 20-task convention used in
  `exps/final_vision/`. Food101 was also moved to 10-task, then brought back
  up to 20-task (2026-07-20) — see note below.
- **Matching methodology corrected 2026-07-20: absolute image-count
  difference, not ratio.** Ratio comparison is scale-biased (e.g. a target at
  40% of a small reference and a target at 40% *above* a large reference can
  have wildly different absolute closeness despite similar-looking ratios).
  All "closest match" calls below use `|target_images_per_task -
  reference_images_per_task|`, not a ratio. This changed one conclusion
  (SUN397, see below) versus the original ratio-based pass; Food101 and
  OmniBenchmark-1K's conclusions were unchanged by the correction.
- **Food101 reverted to 20 tasks** (6/5 split, matching the original
  pre-10-task convention exactly): at 10 tasks Food101 was 7,575 images/task,
  the largest of any dataset in scope and not close to any other dataset's
  current per-task volume. At 20 tasks it's 3,787.5 images/task — nonstandard
  (no other dataset uses a 20-task split with this class/task shape) but the
  closest available match is CIFAR-100's 5,000 images/task (diff 1,213 vs.
  2,584 for ImageNet-R) — closer than the 10-task version was to anything.
  Verified via DataManager: init_cls=6, increment=5, 20 tasks, 101 classes,
  75,750 total images, no remainder.
- **OmniBenchmark-1K is the designated long-horizon evaluation** — deliberately
  the dataset with by far the most tasks (100, vs. 10-20 for everything else).
  **Split finalized as 100 tasks × 10 classes/task** (1000 total training
  images / task-count comparison, verified against the real dataset —
  168,718 train images total): this gives **1,687 images/task**, the closest
  match among all candidates to an existing dataset's current per-task volume
  — specifically ImageNet-R's current 20-task split (1,204 images/task, diff
  483 vs. 3,313 for CIFAR-100). The alternative 50-task/20-class split (3,374
  images/task) was rejected: it sits roughly between CIFAR-100 and ImageNet-R
  without closely matching either. **HP convention: OmniBenchmark-1K should
  use whatever hyperparameters each method already uses for its ImageNet-R
  20-task split** (both the unified-grid batch=48/lr=3e-4 convention, and —
  for TUNA/EASE specifically — the native LAMDA-PILOT ImageNet-R defaults
  documented below; InfLoRA is the one exception, see its own section), as a
  starting point, rather than deriving a new OmniBenchmark-specific HP set
  from scratch. Not yet wired into `scripts/gen_final_vision_configs.py`'s
  `DATASET_CFG` (which currently only covers the 4 main-grid datasets) — open
  question whether this runs as part of the main grid or as a separate
  track, given its distinct purpose.

## Native LAMDA-PILOT Defaults — InfLoRA, TUNA, EASE (+ SeqLoRA)

**Locked in 2026-07-20**, confirmed via a live 3-task smoke test on GPU 1 for
all 5 (method, dataset) pairs with a native config (InfLoRA/CIFAR-100, TUNA/
CIFAR-100, TUNA/ImageNet-R, EASE/CIFAR-100, EASE/ImageNet-R) — all 5 finished
cleanly with sensible, monotonically-decaying CIL accuracy and low-to-modest
forgetting (0.9-6.65 over 3 tasks); no crashes, no anomalies. Metrics JSONs
under `run_logs/final/{inflora,tuna,ease}/metrics_*native_defaults_smoke*.json`.

**Important distinction**: these are LAMDA-PILOT's own *bundled toolkit configs*
(`exps/{inflora,inflora_20t,ease,ease_inr,tuna_cifar,tuna_inr}.json`, shipped
with this repo, unmodified) — a **different source** from each method's own
separate paper/reference-repo defaults (e.g. InfLoRA's own repo uses
epochs=20/batch=128 for CIFAR-100, vs. LAMDA-PILOT's own bundled port below,
which uses epochs=10/batch=48 — already much closer to our unified
convention). Reported separately from the earlier InfLoRA-paper numbers to
avoid conflating the two sources.

**Coverage**: LAMDA-PILOT bundles native configs for CIFAR-100 and ImageNet-R
only, for all three methods. **No bundled config exists for SUN397, Food101,
or OmniBenchmark-1K, for any of InfLoRA/TUNA/EASE** (confirmed by direct
search of `exps/*.json` — none match any of the three model names paired
with those three datasets).

**Extending to SUN397/Food101/OmniBenchmark-1K**: no native config exists for
any of the three on these datasets, so each borrows the *closest-matching*
existing dataset's HPs by **absolute** images/task difference (see
methodology-correction note above): Food101 (3,787.5 img/task) → **CIFAR-100**
(diff 1,213 vs. 2,584); OmniBenchmark-1K (1,687.2 img/task) → **ImageNet-R**
(diff 483 vs. 3,313); SUN397 (1,985.0 img/task) → **ImageNet-R** (diff 781 vs.
3,015) — this one flipped from the original ratio-based pass, which had
called it an ambiguous/weak CIFAR-100 lean; under absolute difference it's a
clear, confident ImageNet-R match instead.

### InfLoRA

**Resolved 2026-07-20**: InfLoRA has no native LAMDA-PILOT config for
ImageNet-R (or anything else besides CIFAR-100), so rather than mixing in a
different-source fallback (its own separate reference-repo ImageNet-R
config), InfLoRA simply **uses its native CIFAR-100 hyperparameters for every
dataset** — CIFAR-100, ImageNet-R, SUN397, Food101, and OmniBenchmark-1K all
identical. This overrides the general "closest images/task match" rule for
InfLoRA specifically, since CIFAR-100 is InfLoRA's only native LAMDA-PILOT
source to begin with.

| | CIFAR-100 (native, `exps/inflora.json`) | ImageNet-R | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native (10-task variant matches our split exactly) | share CIFAR-100 | share CIFAR-100 | share CIFAR-100 | share CIFAR-100 |
| epochs | 10 | 10 | 10 | 10 | 10 |
| optimizer | Adam | Adam | Adam | Adam | Adam |
| lr | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |
| batch size | 48 | 48 | 48 | 48 | 48 |
| lora_rank | 10 | 10 | 10 | 10 | 10 |
| lora_alpha | null (scale×1) | null | null | null | null |
| lamb / lame | 0.95 / 1.0 | 0.95 / 1.0 | 0.95 / 1.0 | 0.95 / 1.0 | 0.95 / 1.0 |

**Correction (2026-07-21)**: this row previously said `40` (scale×4), reflecting
an intended fix from the earlier scaling investigation that was applied to the
*doc* but never actually applied to the config file
(`exps/review/native_defaults_smoke/inflora_cifar224.json`), which has run at
`null` (scale×1) the whole time. Rather than switch the config to match the
stale doc value now, the doc is corrected to match what was actually tested —
keeping InfLoRA on the same scale×1 convention as SeqLoRA/SketchLoRA/O-LoRA/
TreeLoRA (its lr=0.0005/epochs=10/batch=48/lamb/lame stay its own native
values, unaffected). This means every InfLoRA result reported so far (fold-fix
verification's accuracy/persistent-memory numbers, the InfLoRA-vs-SketchLoRA
comparisons) remains valid — no rerun needed.

Note batch=48 in the native CIFAR-100 config already matches our unified grid
convention (unlike InfLoRA's own separate reference repo, which uses 128) —
LAMDA-PILOT's own port had already moved closer to what we use independently.

**Scaling fix (2026-07-20):** `lora_scaling = (lora_alpha if not None else
lora_rank) / lora_rank` (`backbone/vit_lora.py:54`). When rank moved from the
old unified grid's 8 to InfLoRA's native 10, `lora_alpha` was left at `null`,
which silently dropped scaling from the standing ×4 convention (α=32 at rank=8)
to ×1 (α=null at any rank). Fixed by setting `lora_alpha=40` (40/10 = ×4,
preserving the ×4 convention at the new rank) instead of carrying over the
old α=32 (32/10 = ×3.2, wrong) or leaving it null. Applies to every review
smoke config under `exps/review/seqlora_sketchlora_check/` and
`exps/review/sketchlora_lr3e4_check/` — NOT yet reconciled with
`scripts/gen_final_vision_configs.py`, which still uses its own separate
rank=8/α=32 convention (see "Status" section below).

### SketchLoRA

**Decision 2026-07-21**: rank=10, `lora_alpha=null` (scaling ×1 — NOT the
×4 convention InfLoRA/TUNA/EASE use), batch=48 held fixed; lr/epochs/
`svd_energy_target` (ε) tuned per-dataset via an overnight smoke-test search
(smoke size = ⌈tasks/4⌉ — 10-task→3, 20-task→5 — except OmniBenchmark-1K,
capped at a fixed 10-task smoke by explicit instruction rather than the
25 the formula would give, since 100 tasks is too slow to search).

| | CIFAR-100 | ImageNet-R | Food101 | SUN397 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| lr | 0.001 | 0.001 | 0.0005 | 0.001 | 0.001 |
| epochs | 10 | 10 | 10 | 10 | 15 |
| svd_energy_target (ε) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| svd_rank | 10 | 10 | 10 | 10 | 10 |
| lora_rank | 10 | 10 | 10 | 10 | 10 |
| lora_alpha | null | null | null | null | null |
| batch size | 48 | 48 | 48 | 48 | 48 |

Notes:
- CIFAR-100/ImageNet-R (lr=1e-3/10ep/ε=0.01/scale×1) were the pre-existing
  starting point from the same-night scaling investigation (see "Scaling
  fix" note above), not independently re-searched in the overnight run.
- Food101: only lr changed (5e-4 vs the 1e-3 starting point, +2.8 CIL at
  smoke scale) — ε=0.03 alone also helped, but combining lr=5e-4+ε=0.03
  underperformed either change alone (interference, not synergy), so only
  the single best axis was kept.
- SUN397: nothing beat the starting point across 6 tested variants
  (ε∈{0.005,0.01,0.03}, lr∈{5e-4,1e-3,2e-3}, epochs∈{10,15}) — kept as-is.
- OmniBenchmark-1K: only epoch count changed (15 vs 10, +2.4 CIL). epochs=20
  was also tested and performed *worse* than 15, confirming a real peak
  rather than "more epochs is always better."
- All results single-seed (1993), smoke-scale only — not yet confirmed at
  full task count.

### SeqLoRA

**Decision 2026-07-21 (supersedes the 2026-07-20 decision below)**: SeqLoRA
now shares SketchLoRA's hyperparameters (table above) rather than InfLoRA's —
rank=10, `lora_alpha=null` (scale×1), batch=48, and the same per-dataset
lr/epochs (SketchLoRA's `svd_energy_target` has no SeqLoRA equivalent).
Reversed after overnight comparison runs (same seed/splits/smoke sizes) showed
SketchLoRA beating SeqLoRA at every dataset under these settings — Food101
+5.22 CIL, SUN397 +2.92, OmniBenchmark-1K +11.43 (smoke scale) — using
SeqLoRA's own matching lr/epochs, not a mismatched comparison.

*Superseded 2026-07-20 decision, kept for history:* SeqLoRA shared InfLoRA's
hyperparameters — epochs=10, lr=0.0005, batch=48, rank=10, alpha=40
(scaling ×4), identical across all 5 benchmarks.

Note: `models/lora.py` (the shared LoRA-scaffold training loop SeqLoRA/
InfLoRA/O-LoRA/SketchLoRA/TreeLoRA all use) hardcodes AdamW regardless of any
`optim`/`optimizer` config value, so despite InfLoRA's native config
literally saying `"optim": "adam"`, SeqLoRA (like InfLoRA) trains under
AdamW in practice either way — not a real distinction here.

### TreeLoRA

**Decision 2026-07-21**: TreeLoRA shares SketchLoRA/SeqLoRA's hyperparameters
— rank=10, `lora_alpha=null` (scale×1), batch=48, and the same per-dataset
lr/epochs (table above) — plus its own `reg=0.1` (corrected this session from
0.5, which traced to the reference repo's NLP/TRACE launch script rather than
the paper's own vision-section default; see the "Native Paper/Repo Defaults"
section below for the discrepancy detail).

This was cross-checked against TreeLoRA's own paper-native settings (batch=192,
constant lr=0.005, 20 epochs CIFAR-100/50 elsewhere, same reg=0.1) run
head-to-head at smoke scale, same seed: the native settings underperformed our
tuned convention by a wide margin on CIFAR-100 (81.2 vs 92.37 final-task CIL,
-11.2), so the native-config comparison was cut short (ImageNet-R/SUN397/
OmniBenchmark-1K not run) and the SketchLoRA-matching settings were locked in
instead.

Result at these settings (smoke scale, single seed 1993): CIFAR-100 92.37,
ImageNet-R 81.98, Food101 **86.88** (best Food101 result of any method
tested), SUN397 77.85, OmniBenchmark-1K 75.58 — competitive with or ahead of
SketchLoRA/O-LoRA on 4 of 5 datasets, clearly ahead of SeqLoRA on all 5.

### TUNA

**Decision 2026-07-21 (reverts both the 2026-07-20 batch-unification AND
epoch-unification below)**: batch size AND epoch count both reverted back to
the pristine native values (batch 16/CIFAR-100, 32/ImageNet-R; epochs
15/CIFAR-100, 10/ImageNet-R unchanged). Neither the batch=48 nor the
epochs=10-on-CIFAR-100 change was ever actually applied to the source config
files (`exps/tuna_cifar.json` still held batch=16/epochs=15 the whole time),
so this revert is a doc-only change, nothing to undo in the actual configs.
Propagated to both borrowed columns (Food101 borrows CIFAR-100,
SUN397/OmniBenchmark-1K borrow ImageNet-R) automatically.

| | CIFAR-100 (`exps/tuna_cifar.json`) | ImageNet-R (`exps/tuna_inr.json`) | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native (fully unchanged) | native (fully unchanged) | borrow ImageNet-R | borrow CIFAR-100 | borrow ImageNet-R |
| epochs | 15 | 10 | 10 | 15 | 10 |
| optimizer | SGD | SGD | SGD | SGD | SGD |
| lr | 0.01 | 0.02 | 0.02 | 0.01 | 0.02 |
| batch size | 16 | 32 | 32 | 16 | 32 |
| r (adapter rank) | 16 | 16 | 16 | 16 | 16 |
| use_orth | false | true | true | false | true |
| decay | false | true | true | false | true |

No LR-scaling caveat needed now — with batch reverted to native, there's no
batch change to reconcile lr against.

Native splits differ from our own chosen splits (CIFAR-100: native 20 tasks
vs. our 10; ImageNet-R: native 10 tasks vs. our 20) — HPs reported as-is from
the native config regardless, since split and per-step HPs are tracked
separately in this document.

### EASE

**Decision 2026-07-21 (reverts the 2026-07-20 batch-unification below)**:
ImageNet-R's batch size reverted back to its pristine native value (16) — as
with TUNA, `exps/ease_inr.json` itself was never actually changed to 48, so
this is a doc-only revert. CIFAR-100 was already 48 natively, unaffected.
Epoch counts unchanged throughout (20 everywhere). Propagated to
SUN397/OmniBenchmark-1K (both borrow ImageNet-R) automatically.

| | CIFAR-100 (`exps/ease.json`) | ImageNet-R (`exps/ease_inr.json`) | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native (unchanged) | native (unchanged) | borrow ImageNet-R | borrow CIFAR-100 | borrow ImageNet-R |
| epochs | 20 | 20 | 20 | 20 | 20 |
| optimizer | SGD | SGD | SGD | SGD | SGD |
| lr | 0.025 | 0.05 | 0.05 | 0.025 | 0.05 |
| batch size | 48 | 16 | 16 | 48 | 16 |
| weight_decay | 0.0005 | 0.005 | 0.005 | 0.0005 | 0.005 |
| ffn_num (adapter dim) | 64 | 64 | 64 | 64 | 64 |
| prompt_token_num | 5 | 5 | 5 | 5 | 5 |
| alpha | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

No LR-scaling caveat needed now, for the same reason as TUNA above.

Same split caveat as TUNA above. ffn_num=64 here is notably larger than the
rank-8 convention used elsewhere in our grid — EASE's adapter dimension isn't
directly comparable to a LoRA rank, but this is a real capacity difference
worth being aware of if adopting these native settings as-is.

## Native Paper/Repo Defaults — CL-LoRA, TreeLoRA

Distinct from the "Native LAMDA-PILOT Defaults" section above: these come
from each method's *own* paper (`literature/CL-LoRA.pdf`, `literature/
treelora.txt`) and its own separate reference repo (`CL-LoRA/`, `TreeLoRA/`),
not a LAMDA-PILOT-bundled config — there is no LAMDA-PILOT-native config for
either method on any of our 5 benchmarks.

**Coverage**: neither paper evaluates on SUN397, Food101, or OmniBenchmark —
confirmed by full-text search of both papers (zero mentions of any of the
three) and by their reference repos (CL-LoRA's `README.md`/`exps/` only list
CIFAR-100/ImageNet-R/ImageNet-A/VTAB; TreeLoRA's repo contains *no vision
code at all*, only the NLP/TRACE side — its vision numbers come from the
paper text alone). Explicit non-coverage, not an unfound gap.

### CL-LoRA

Evaluates CIFAR-100, ImageNet-R, ImageNet-A, VTAB (paper Sec 5.1: *"we
conduct comprehensive experiments on four representative CIL benchmarks
including CIFAR-100, ImageNet-R, ImageNet-A, and VTAB"*). lr/epochs/batch/
optimizer below come from the repo's own `exps/*.json` (not stated per-
dataset in the paper text, which only fixes rank=10, λ1=5, λ2=0.0001, split
point l=6 as constants across all its datasets, Sec 5.1 Implementation
Details).

| | CIFAR-100 (`CL-LoRA/exps/cifar.json`) | ImageNet-R (`CL-LoRA/exps/inr.json`) | SUN397 | Food101 | OmniBenchmark |
|---|---|---|---|---|---|
| native task split | 20 tasks × 5 cls (**not** our 10×10) | 40 tasks × 5 cls (**not** our 20×10) | *not evaluated* | *not evaluated* | *not evaluated* |
| epochs | 30 / 30 | 20 / 20 | — | — | — |
| optimizer | SGD, cosine | SGD, cosine | — | — | — |
| lr | 0.03 | 0.05 | — | — | — |
| batch size | 64 | 32 | — | — | — |
| rank | 10 | 10 | — | — | — |
| λ1 / λ2 | 5 / 0.0001 | 5 / 0.0001 | — | — | — |

Both native splits are considerably finer-grained than ours (more, smaller
tasks) — a real mismatch worth being aware of if adopting these HPs, since
epoch/lr choices tuned for 20-40 tasks may not transfer cleanly to our
10-20-task splits.

### TreeLoRA

Vision-track benchmarks: Split CIFAR-100, Split ImageNet-R, Split CUB-200
(paper Sec 5.1; CUB-200 out of scope here). No SUN397/Food101/OmniBenchmark
mentions anywhere in the paper or repo.

| | CIFAR-100 | ImageNet-R | SUN397 | Food101 | OmniBenchmark |
|---|---|---|---|---|---|
| native task split | 10 tasks × 10 cls (**matches our split exactly**) | 10 tasks × 20 cls (**not** our 20×10) | *not evaluated* | *not evaluated* | *not evaluated* |
| epochs | 20 | 50 | — | — | — |
| optimizer | Adam (β1=0.9, β2=0.999) | Adam | — | — | — |
| lr | 0.005 | 0.005 | — | — | — |
| batch size | 192 | 192 | — | — | — |
| λ (reg coeff) | 0.1 | 0.1 (no per-dataset override stated) | — | — | — |

CIFAR-100's native split matches ours exactly, unlike CL-LoRA's.

**Confirmed discrepancy** (verified directly, not just from the porting
comment): our actual configs — `models/treelora.py`'s own default
(`self.reg = args.get("reg", 0.5)`), `exps/review/task_incremental_imr5t/
treelora.json`, and every `exps/final_vision/*_treelora_*.json` — all use
`reg=0.5`. That value traces back to `TreeLoRA/scripts/lora_based_methods/
Tree_LoRA.sh`, which is the **NLP/TRACE launch script**, not a vision source.
The paper's own vision-section default is **λ=0.1** (Table 6 sensitivity
study) — we're currently running TreeLoRA at 5x its paper-recommended
regularization strength for vision. Worth deciding whether to correct this.

## Status: not yet applied to the production pipeline

Everything in this document (dataset splits, InfLoRA/TUNA/EASE/SeqLoRA
baseline HPs) is a **decision record only** — `scripts/gen_final_vision_configs.py`
and `exps/final_vision/*.json` (the configs the live cluster run is actually
using) still reflect the *earlier* convention: 20-task splits for CIFAR-100/
ImageNet-R/Food101/SUN397 (the old init_cls/increment values, not the ones
in the Dataset Splits table above), OmniBenchmark-1K not present at all, and
the unified batch=48/lr=3e-4/rank=8 convention applied uniformly to all 10
methods including InfLoRA/TUNA/EASE/SeqLoRA (not their native per-dataset
baselines above). None of this document's decisions have been wired into the
actual generator or regenerated yet.
