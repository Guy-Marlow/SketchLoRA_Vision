"""SeqLoRA (sequential finetuning) at lora_rank=20, across all five
wave1_final master-table datasets, 3 seeds each (2026-11-03 user request).

Base configs are each dataset's own exps/wave1_final/seqlora_{dataset}_
s{seed}.json -- the project's standing rank-10 SeqLoRA reference -- with
ONLY lora_rank changed (10 -> 20) and a distinct prefix (so this doesn't
collide with the rank-10 run's own done-tag in run_logs/final/seqlora/,
which the "{model}_{dataset}_{prefix}_s{seed}" tag scheme keys off of).
Every other field (init_lr, init_cls/increment, tuned_epoch, batch_size,
weight_decay, lora_merge, etc.) is left exactly as wave1_final set it per
dataset -- this is a rank ablation, not a fresh tune.

5 datasets x 3 seeds = 15 runs.
"""
import json
import os

OUT_DIR = "exps/seqlora_rank20_wave1"
SEEDS = [1993, 1996, 1999]
DATASETS = ["imagenetr", "cifar224", "sun397", "food101", "omnibenchmark1k"]
BASE_CONFIG = "exps/wave1_final/seqlora_{dataset}_s{seed}.json"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset in DATASETS:
        for seed in SEEDS:
            base_path = BASE_CONFIG.format(dataset=dataset, seed=seed)
            cfg = json.load(open(base_path))
            cfg["lora_rank"] = 20
            cfg["prefix"] = "seqlora_rank20_wave1_{}_s{}".format(dataset, seed)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)

    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
