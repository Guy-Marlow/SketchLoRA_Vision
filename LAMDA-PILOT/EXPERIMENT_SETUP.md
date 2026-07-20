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
| Food101 | 101 | 11 | 10 | 10 |
| OmniBenchmark-1K | 1000 | 10 | 10 | 100 |

Notes:
- SUN397 (397) is prime, so no split of 10 tasks can be perfectly even; 37/40
  (task 0 slightly smaller, the other 9 equal) was chosen over the alternative
  10/43 to keep the non-uniform task closer in size to the rest.
- CIFAR-100, Food101 (101 = 11 + 9×10), and OmniBenchmark-1K (1000 = 100×10,
  perfectly even) all divide cleanly or near-cleanly.
- OmniBenchmark-1K is back in scope as of 2026-07-20 (previously dropped
  2026-07-19 for being under-studied/longest-running); ImageNet-R's split is
  unchanged from the prior convention. CIFAR-100/SUN397/Food101 all move from
  20-task to 10-task splits, replacing the earlier 20-task convention used in
  `exps/final_vision/`.
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

### InfLoRA

| | CIFAR-100 (native split: 10 tasks, `exps/inflora.json`) | CIFAR-100 (native split: 20 tasks, `exps/inflora_20t.json`) | ImageNet-R |
|---|---|---|---|
| init_cls/increment | 10/10 | 5/5 | *no bundled LAMDA-PILOT config* |
| epochs | 10 | 10 | — |
| optimizer | Adam | Adam | — |
| lr | 0.0005 | 0.0005 | — |
| batch size | 48 | 48 | — |
| lora_rank | 10 | 10 | — |
| lamb / lame | 0.95 / 1.0 | 0.95 / 1.0 | — |

The 10-task variant's split (10/10) matches our own chosen CIFAR-100 split
exactly. Note batch=48 here already matches our unified convention (unlike
InfLoRA's own separate reference repo, which uses 128) — LAMDA-PILOT's own
port had already moved closer to what we're using independently.

### TUNA

| | CIFAR-100 (`exps/tuna_cifar.json`, 20-task: 5/5) | ImageNet-R (`exps/tuna_inr.json`, 10-task: 20/20) |
|---|---|---|
| epochs | 15 | 10 |
| optimizer | SGD | SGD |
| lr | 0.01 | 0.02 |
| batch size | 16 | 32 |
| r (adapter rank) | 16 | 16 |
| use_orth | false | true |
| decay | false | true |

Both native splits differ from our own chosen splits (CIFAR-100: native 20
tasks vs. our 10; ImageNet-R: native 10 tasks vs. our 20) — HPs reported
as-is from the native config regardless, since split and per-step HPs are
tracked separately in this document.

### EASE

| | CIFAR-100 (`exps/ease.json`, 20-task: 5/5) | ImageNet-R (`exps/ease_inr.json`, 10-task: 20/20) |
|---|---|---|
| epochs | 20 | 20 |
| optimizer | SGD | SGD |
| lr | 0.025 | 0.05 |
| batch size | 48 | 16 |
| weight_decay | 0.0005 | 0.005 |
| ffn_num (adapter dim) | 64 | 64 |
| prompt_token_num | 5 | 5 |
| alpha | 0.1 | 0.1 |

Same split caveat as TUNA above. ffn_num=64 here is notably larger than the
rank-8 convention used elsewhere in our grid — EASE's adapter dimension isn't
directly comparable to a LoRA rank, but this is a real capacity difference
worth being aware of if adopting these native settings as-is.
