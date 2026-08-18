"""LR tuning sweep for SketchLoRA's fixed-rank variant (2026-08-18 user request):
motivated by the observation that O-LoRA's orthogonality penalty effectively acts
as a brake on the live slot's updates, raising the question of whether SketchLoRA
is simply under/mistuned on its learning rate -- this sweep answers that directly
for the fixed-rank config specifically (no adaptive rank growth), across 3
benchmarks, at 6 learning rates x 2 seeds each.

Base template is exps/sketchlora_ablations_imagenetr20t/sketchlora_fixedrank_s1993.json
verbatim -- randSVD merge_op, fixed compression to rank 10 (svd_rank=10, no
svd_energy_target field, sketchlora_admission="global_eps", matching that file's
own "no adaptive growth" convention), batch_size=48, tuned_epoch=20, weight_decay
0.0005 on the head + currently-training adapter (sketchlora_lora_wd=0.0 zeroes it
for the sketch specifically) -- ALL unchanged from that file, per explicit user
instruction ("everything else should be set the way it was prior"). Only
dataset/init_cls/increment/stop_after_tasks/seed/init_lr/prefix vary per cell.

final_metrics/print_forget both OFF, matching the project's other pure LR-sweep
convention (scripts/gen_lr_sweep_configs.py): this sweep only needs each cell's
final top-5 CIL, which trainer.py prints unconditionally ("CNN top5 curve:" /
"Average Accuracy (CNN):"), so there's no need for MetricsLogger/CE-ledger
overhead here.

Task counts (explicit user spec, truncating each benchmark's full oracle split
via stop_after_tasks): CIFAR-100-10t -> 5, ImageNet-R-20t -> 10,
OmniBenchmark-1k-100t -> 20.
"""
import json
import os

BASE_TEMPLATE = "exps/sketchlora_ablations_imagenetr20t/sketchlora_fixedrank_s1993.json"
OUT_DIR = "exps/sketchlora_fixedrank_lr_tune"
SEEDS = [1993, 1996]

# (dataset, init_cls, increment, stop_after_tasks) -- init_cls/increment match
# the wave1_final convention for each dataset exactly; stop_after_tasks per the
# user's explicit counts above.
DATASETS = [
    ("cifar224", 10, 10, 5),
    ("imagenetr", 10, 10, 10),
    ("omnibenchmark1k", 10, 10, 20),
]

# (tag, value) -- order matches the user's own listing.
LEARNING_RATES = [
    ("1e-2", 1e-2), ("3e-3", 3e-3), ("1e-3", 1e-3),
    ("8e-4", 8e-4), ("3e-4", 3e-4), ("1e-4", 1e-4),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = json.load(open(BASE_TEMPLATE))
    written = []
    for dataset, init_cls, increment, stop_after_tasks in DATASETS:
        for lr_tag, lr in LEARNING_RATES:
            for seed in SEEDS:
                cfg = dict(base)
                cfg["dataset"] = dataset
                cfg["init_cls"] = init_cls
                cfg["increment"] = increment
                cfg["stop_after_tasks"] = stop_after_tasks
                cfg["seed"] = [seed]
                cfg["init_lr"] = lr
                cfg["final_metrics"] = False
                cfg["print_forget"] = False
                cfg["prefix"] = "sketchlora_fixedrank_lr_tune_{}_{}_s{}".format(dataset, lr_tag, seed)
                out = os.path.join(OUT_DIR, "sketchlora_{}_{}_s{}.json".format(dataset, lr_tag, seed))
                json.dump(cfg, open(out, "w"), indent=2)
                written.append(out)
    print("wrote {} configs to {}/ ({} datasets x {} lrs x {} seeds)".format(
        len(written), OUT_DIR, len(DATASETS), len(LEARNING_RATES), len(SEEDS)))


if __name__ == "__main__":
    main()
