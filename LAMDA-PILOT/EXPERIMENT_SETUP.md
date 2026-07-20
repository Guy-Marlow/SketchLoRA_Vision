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
| OmniBenchmark-1K | 1000 | 20 | 20 | 50 |

Notes:
- SUN397 (397) is prime, so no split of 10 tasks can be perfectly even; 37/40
  (task 0 slightly smaller, the other 9 equal) was chosen over the alternative
  10/43 to keep the non-uniform task closer in size to the rest.
- CIFAR-100, Food101 (101 = 11 + 9×10), and OmniBenchmark-1K (1000 = 50×20,
  perfectly even) all divide cleanly or near-cleanly.
- OmniBenchmark-1K is back in scope as of 2026-07-20 (previously dropped
  2026-07-19 for being under-studied/longest-running); ImageNet-R's split is
  unchanged from the prior convention. CIFAR-100/SUN397/Food101 all move from
  20-task to 10-task splits, replacing the earlier 20-task convention used in
  `exps/final_vision/`.
