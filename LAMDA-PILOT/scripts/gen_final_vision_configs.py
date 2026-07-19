"""Generate the final headline vision-experiment configs: 9 methods x 4 datasets
x 3 seeds, full (non-truncated) 20-task splits, pure task-boundary training
(NOT the memory-increment streaming design -- that's a separate track).

HP convention (user-confirmed 2026-07-19, live-validated via two 36-run 3-task
sanity sweeps): batch_size=48, lr=3e-4, applied uniformly to all 9 methods,
including RainbowPrompt/ProgPrompt (a deliberate departure from their own
reference-paper LR/batch values, in favor of one shared, tested convention
across the whole roster). LoRA rank stays 8 for the LoRA-scaffold family;
EASE/TUNA's own adapter rank (ffn_num/r) was separately set to 8 already in
their HP-source configs (exps/review/task_incremental_imr5t/{ease,tuna}.json).

"final_metrics": true on every config -- see utils/metrics_logger.py +
utils/flops.py for what gets recorded (persistent_state/_deployed_forward now
implemented for all 9 methods).

HiDeLoRA and OmniBenchmark-1k are OUT of scope per user decision 2026-07-19
(HiDeLoRA won't be used in either final evaluation; OmniBenchmark-1k dropped
for being under-studied and by far the longest-running split).
"""
import json
import os

SRC_DIR = "exps/review/task_incremental_imr5t"
OUT_DIR = "exps/final_vision"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "treelora",
           "rainbowprompt", "progprompt", "ease", "tuna"]
SEEDS = [1993, 1994, 1995]

# Full 20-task splits (established convention, see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md
# and the plan doc's task-split discussion -- sun397/food101 don't divide evenly into 20
# equal tasks, so init_cls absorbs the remainder).
DATASET_CFG = {
    "cifar224": dict(init_cls=5, increment=5),
    "imagenetr": dict(init_cls=10, increment=10),
    "food101": dict(init_cls=6, increment=5),
    "sun397": dict(init_cls=17, increment=20),
}

BATCH_SIZE = 48
LR = 3e-4

os.makedirs(OUT_DIR, exist_ok=True)
written = []
for dataset, dcfg in DATASET_CFG.items():
    for m in METHODS:
        src = os.path.join(SRC_DIR, f"{m}.json")
        base = json.load(open(src))
        for seed in SEEDS:
            d = dict(base)
            d["dataset"] = dataset
            d["init_cls"] = dcfg["init_cls"]
            d["increment"] = dcfg["increment"]
            d.pop("stop_after_tasks", None)          # full run, no truncation
            d.pop("boundary_mode", None)              # pure task-boundary track
            d.pop("stream_budget_mb", None)
            d["batch_size"] = BATCH_SIZE
            d["init_lr"] = LR
            if "later_lr" in d:
                d["later_lr"] = LR
            d["seed"] = [seed]
            d["final_metrics"] = True
            d["print_forget"] = True
            # Every SLURM array task gets its own single-GPU allocation, which
            # appears as device 0 from inside that job (CUDA_VISIBLE_DEVICES is
            # set by SLURM/the scheduler to restrict visibility) -- NOT the
            # multi-GPU manual device-index pinning scripts/run_final_vision_
            # experiments.sh's bash launcher uses. Fine for either launch path:
            # the bash launcher overwrites this per-lane anyway.
            d["device"] = ["0"]
            tag = f"{dataset}_{m}_s{seed}"
            d["prefix"] = f"final_vision_{tag}"
            path = os.path.join(OUT_DIR, f"{tag}.json")
            json.dump(d, open(path, "w"), indent=2)
            written.append(tag)

print(f"wrote {len(written)} configs to {OUT_DIR}/ "
      f"({len(DATASET_CFG)} datasets x {len(METHODS)} methods x {len(SEEDS)} seeds)")
