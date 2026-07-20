# Experimental Setup

Living document describing the datasets, task splits, and per-method
hyperparameters used in the final vision continual-learning evaluation.
Built up incrementally; sections are added as they're settled.

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
- **Food101 reverted to 20 tasks** (6/5 split, matching the original
  pre-10-task convention exactly): at 10 tasks Food101 was 7,575 images/task,
  the largest of any dataset in scope and not close to any other dataset's
  current per-task volume. At 20 tasks it's 3,787.5 images/task — nonstandard
  (no other dataset uses a 20-task split with this class/task shape) but the
  closest available match is CIFAR-100's 5,000 images/task (ratio 0.76x,
  diff 1,213) — closer than the 10-task version was to anything. Verified via
  DataManager: init_cls=6, increment=5, 20 tasks, 101 classes, 75,750 total
  images, no remainder.
- **OmniBenchmark-1K is the designated long-horizon evaluation** — deliberately
  the dataset with by far the most tasks (100, vs. 10-20 for everything else).
  **Split finalized as 100 tasks × 10 classes/task** (1000 total training
  images / task-count comparison, verified against the real dataset —
  168,718 train images total): this gives **1,687 images/task**, the closest
  match among all candidates to an existing dataset's current per-task volume
  — specifically ImageNet-R's current 20-task split (1,204 images/task, ratio
  1.40x). The alternative 50-task/20-class split (3,374 images/task) was
  rejected: it sits roughly between CIFAR-100 (5,000/task, ratio 0.67x) and
  ImageNet-R (ratio 2.80x) without closely matching either. **HP convention:
  OmniBenchmark-1K should use whatever hyperparameters each method already
  uses for its ImageNet-R 20-task split** (both the unified-grid batch=48/
  lr=3e-4 convention, and — for InfLoRA/TUNA/EASE specifically — the native
  LAMDA-PILOT ImageNet-R defaults documented below), as a starting point,
  rather than deriving a new OmniBenchmark-specific HP set from scratch. Not
  yet wired into `scripts/gen_final_vision_configs.py`'s `DATASET_CFG` (which
  currently only covers the 4 main-grid datasets) — open question whether
  this runs as part of the main grid or as a separate track, given its
  distinct purpose.

## Native LAMDA-PILOT Defaults — InfLoRA, TUNA, EASE

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
existing dataset's HPs by images/task (same principle used for the dataset-
split decisions above): Food101 (3,787.5 img/task) → **CIFAR-100** (ratio
0.76x); OmniBenchmark-1K (1,687.2 img/task) → **ImageNet-R** (ratio 1.40x);
SUN397 (1,985.0 img/task) is a **weak, ambiguous call** — technically closest
to CIFAR-100 (ratio 0.40x from 1) vs. ImageNet-R (ratio 0.65x from 1), but the
margin is small enough that this shouldn't be treated as a confident match
either way; flagged rather than silently resolved.

### InfLoRA

| | CIFAR-100 (native, `exps/inflora.json`) | ImageNet-R | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native (10-task variant matches our split exactly) | **no native LAMDA-PILOT config at all** | borrow CIFAR-100 (weak match) | borrow CIFAR-100 | borrow ImageNet-R — **but InfLoRA has no native ImageNet-R HPs to borrow** |
| epochs | 10 | ? | 10 | 10 | ? |
| optimizer | Adam | ? | Adam | Adam | ? |
| lr | 0.0005 | ? | 0.0005 | 0.0005 | ? |
| batch size | 48 | ? | 48 | 48 | ? |
| lora_rank | 10 | ? | 10 | 10 | ? |
| lamb / lame | 0.95 / 1.0 | ? | 0.95 / 1.0 | 0.95 / 1.0 | ? |

**Open gap**: InfLoRA has no bundled LAMDA-PILOT ImageNet-R config at all, and
OmniBenchmark-1K's convention is "borrow ImageNet-R" — so InfLoRA has nothing
to borrow for either. Two options: (a) fall back to InfLoRA's own *separate*
reference-repo ImageNet-R config (`mimg20_inflora.json`, reported earlier in
conversation: epochs=50, batch=128, lr=0.0005, rank=10, lamb=0.98 — a
different source than "LAMDA-PILOT native," consistent with how this
document has been distinguishing the two sources throughout), or (b) borrow
CIFAR-100's InfLoRA HPs for both ImageNet-R and OmniBenchmark-1K instead.
Needs a decision — not resolved yet.

Note batch=48 in the native CIFAR-100 config already matches our unified grid
convention (unlike InfLoRA's own separate reference repo, which uses 128) —
LAMDA-PILOT's own port had already moved closer to what we use independently.

### TUNA

| | CIFAR-100 (native, `exps/tuna_cifar.json`) | ImageNet-R (native, `exps/tuna_inr.json`) | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native | native | borrow CIFAR-100 (weak match) | borrow CIFAR-100 | borrow ImageNet-R |
| epochs | 15 | 10 | 15 | 15 | 10 |
| optimizer | SGD | SGD | SGD | SGD | SGD |
| lr | 0.01 | 0.02 | 0.01 | 0.01 | 0.02 |
| batch size | 16 | 32 | 16 | 16 | 32 |
| r (adapter rank) | 16 | 16 | 16 | 16 | 16 |
| use_orth | false | true | false | false | true |
| decay | false | true | false | false | true |

Native splits differ from our own chosen splits (CIFAR-100: native 20 tasks
vs. our 10; ImageNet-R: native 10 tasks vs. our 20) — HPs reported as-is from
the native config regardless, since split and per-step HPs are tracked
separately in this document.

### EASE

| | CIFAR-100 (native, `exps/ease.json`) | ImageNet-R (native, `exps/ease_inr.json`) | SUN397 | Food101 | OmniBenchmark-1K |
|---|---|---|---|---|---|
| source | native | native | borrow CIFAR-100 (weak match) | borrow CIFAR-100 | borrow ImageNet-R |
| epochs | 20 | 20 | 20 | 20 | 20 |
| optimizer | SGD | SGD | SGD | SGD | SGD |
| lr | 0.025 | 0.05 | 0.025 | 0.025 | 0.05 |
| batch size | 48 | 16 | 48 | 48 | 16 |
| weight_decay | 0.0005 | 0.005 | 0.0005 | 0.0005 | 0.005 |
| ffn_num (adapter dim) | 64 | 64 | 64 | 64 | 64 |
| prompt_token_num | 5 | 5 | 5 | 5 | 5 |
| alpha | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

Same split caveat as TUNA above. ffn_num=64 here is notably larger than the
rank-8 convention used elsewhere in our grid — EASE's adapter dimension isn't
directly comparable to a LoRA rank, but this is a real capacity difference
worth being aware of if adopting these native settings as-is.

## Status: not yet applied to the production pipeline

Everything in this document (dataset splits, InfLoRA/TUNA/EASE baseline HPs)
is a **decision record only** — `scripts/gen_final_vision_configs.py` and
`exps/final_vision/*.json` (the configs the live cluster run is actually
using) still reflect the *earlier* convention: 20-task splits for CIFAR-100/
ImageNet-R/Food101/SUN397 (the old init_cls/increment values, not the ones
in the Dataset Splits table above), OmniBenchmark-1K not present at all, and
the unified batch=48/lr=3e-4/rank=8 convention applied uniformly to all 10
methods including InfLoRA/TUNA/EASE (not their native per-dataset baselines
above). None of this document's decisions have been wired into the actual
generator or regenerated yet.
