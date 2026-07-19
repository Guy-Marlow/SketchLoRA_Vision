"""Generate the final headline vision-experiment configs: 10 methods x 4 datasets
x 3 seeds, full (non-truncated) 20-task splits, pure task-boundary training
(NOT the memory-increment streaming design -- that's a separate track).

HP convention (user-confirmed 2026-07-19, live-validated via two 36-run 3-task
sanity sweeps): batch_size=48, lr=3e-4, applied uniformly to all 10 methods,
including RainbowPrompt/ProgPrompt (a deliberate departure from their own
reference-paper LR/batch values, in favor of one shared, tested convention
across the whole roster). LoRA rank stays 8 for the LoRA-scaffold family;
EASE/TUNA's own adapter rank (ffn_num/r) was separately set to 8 already in
their HP-source configs (exps/review/task_incremental_imr5t/{ease,tuna}.json).

ProgPrompt epoch count: bumped from its native 5 to 15 (user decision
2026-07-19) after a 3-task ablation at 5/10/15 epochs on ImageNet-R showed a
clean monotonic improvement in per-task accuracy with more training (68.35 ->
80.06 -> 84.49 on task 0), i.e. 5 epochs was undertraining it under the
shared batch=48/lr=3e-4 convention. EPOCH_OVERRIDES below applies this to
ProgPrompt's "tuned_epoch" field only -- no other method's epoch count changed.
(Caveat: CIFAR-100 does NOT follow the same trend -- 15 epochs there is worse
than 10 on tasks 1-2 and has the highest forgetting of the three settings,
likely overfitting/interference specific to CIFAR's short 5-class tasks. Kept
at 15 uniformly per user decision, flagged here as a known result caveat.)

CL-LoRA (added to the roster 2026-07-19): its HP-source config used SGD/lr=0.03
natively (exps/review/task_incremental_imr5t/cllora.json), the only method in
the grid not on an Adam-family optimizer. A quick 2-task ablation at the forced
batch=48/lr=3e-4 convention showed SGD works but noticeably underperforms
(82.3/74.5 task0/task1 acc) vs switching to AdamW at the same LR (93.4/88.1) --
3e-4 is an Adam-scale LR, ~100x too low for un-adapted SGD. Fixed by changing
cllora.json's "optimizer" field to "adamw" (matches every other method in the
grid); epoch count (30/task, native) and ffn_num=8 (already rank-8) left as-is.

"final_metrics": true on every config -- see utils/metrics_logger.py +
utils/flops.py for what gets recorded (persistent_state/_deployed_forward now
implemented for all 10 methods).

HiDeLoRA and OmniBenchmark-1k are OUT of scope per user decision 2026-07-19
(HiDeLoRA won't be used in either final evaluation; OmniBenchmark-1k dropped
for being under-studied and by far the longest-running split).
"""
import json
import os

SRC_DIR = "exps/review/task_incremental_imr5t"
OUT_DIR = "exps/final_vision"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "treelora", "cllora",
           "rainbowprompt", "progprompt", "ease", "tuna"]
SEEDS = [1993, 1994, 1995]

# method -> {config_field: value} overrides applied after the batch/lr convention.
EPOCH_OVERRIDES = {
    "progprompt": {"tuned_epoch": 15},
}

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
            d.update(EPOCH_OVERRIDES.get(m, {}))
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
