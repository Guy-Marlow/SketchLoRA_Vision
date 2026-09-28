"""Build two SketchLoRA (orth) configs per (dataset, seed) whose task order is
derived from the SPOT similarity matrix in exps/spot_splits_fast/<dataset>_s<seed>.json
(the 100-epoch/patience-10/runs-3 "wave2_fast" SPOT campaign, ImageNet-R-20t and
CIFAR-100/cifar224-10t) instead of the usual shuffle+seed permutation -- one
with adjacent tasks maximally similar, one maximally dissimilar. Both configs
are the baseline exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json
loaded verbatim, with only shuffle/class_order/prefix overridden (see
utils/data_manager.py's class_order override).

Generalized from scripts/gen_spot_order_configs.py (ImageNet-R-only, reads
exps/spot_splits/ -- the original 200-epoch/patience-15/runs-5 SPOT data) to
take --dataset and read from exps/spot_splits_fast/ instead. Total class
count is derived from the SPOT file's own class_order length rather than
hardcoded, so it works for both ImageNet-R (200 classes) and CIFAR-100 (100
classes) without a dataset-specific constant.

Ordering is a greedy nearest-neighbor chain over sim_matrix_symmetric: start
from the most (least) similar pair, then repeatedly append whichever
remaining task is most (least) similar to the last task placed. Each task
keeps its own class set -- only the task SEQUENCE changes, not which classes
belong to which task.

Usage: python scripts/gen_spot_order_wave2_fast_configs.py --dataset imagenetr --seed 1996
       python scripts/gen_spot_order_wave2_fast_configs.py --dataset cifar224 --seed 1996
"""
import argparse
import json
import os

SPOT_DIR = "exps/spot_splits_fast"
BASELINE_DIR = "exps/sketchlora_orth_wave1_datasets"
OUT_DIR = "exps/spot_order_wave2_fast"


def greedy_chain(sim, maximize):
    n = len(sim)
    pick = max if maximize else min
    start = pick(((a, b) for a in range(n) for b in range(n) if a != b),
                 key=lambda ab: sim[ab[0]][ab[1]])
    order = [start[0], start[1]]
    unvisited = set(range(n)) - {start[0], start[1]}
    while unvisited:
        last = order[-1]
        nxt = pick(unvisited, key=lambda j: sim[last][j])
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["imagenetr", "cifar224"])
    parser.add_argument("--seed", type=int, required=True)
    cli = parser.parse_args()

    spot_path = os.path.join(SPOT_DIR, "{}_s{}.json".format(cli.dataset, cli.seed))
    baseline_config = os.path.join(BASELINE_DIR, "{}_s{}.json".format(cli.dataset, cli.seed))

    spot = json.load(open(spot_path))
    assert spot["seed"] == cli.seed
    assert spot["dataset"] == cli.dataset
    sim = spot["sim_matrix_symmetric"]
    task_classes = spot["task_classes"]
    n = len(task_classes)
    n_classes_total = len(spot["class_order"])

    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, maximize in [("similar", True), ("dissimilar", False)]:
        task_order = greedy_chain(sim, maximize)
        class_order = [c for t in task_order for c in task_classes[t]]
        assert sorted(class_order) == list(range(n_classes_total)), \
            "class_order must be a permutation of 0..{}".format(n_classes_total - 1)

        cfg = json.load(open(baseline_config))
        cfg["shuffle"] = False
        cfg["class_order"] = class_order
        cfg["prefix"] = "spot_order_wave2_fast_{}_{}_s{}".format(tag, cli.dataset, cli.seed)
        out_path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(cli.dataset, tag, cli.seed))
        json.dump(cfg, open(out_path, "w"), indent=2)
        print("wrote", out_path)
        print("  task_order:", task_order)
        adjacent_sims = [sim[task_order[i]][task_order[i + 1]] for i in range(n - 1)]
        print("  mean adjacent similarity: {:.4f}".format(sum(adjacent_sims) / len(adjacent_sims)))


if __name__ == "__main__":
    main()
