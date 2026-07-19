"""Generate boundary-agnostic (sample-driven, memory-constraint-clocked) streaming
smoke configs for the 8-method roster, reusing each method's already-finalized
task-incremental HP conventions (exps/review/task_incremental_imr5t/*.json as the
HP source of truth, minus dataset/task-split fields which get overridden per the
new per-dataset task splits computed in BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md).

**Memory-constraint convention (user-confirmed 2026-07-19): flat 250MB/500MB
across every dataset** -- supersedes the earlier per-dataset 1.5x-mean-samples/
task values (538.33/259.22/etc.), which were only ever a smoke-test placeholder.
This matches the SAME 250/500MB convention already settled for the (now-
superseded) BudgetStreamManager design, now confirmed to carry over here too.
"""
import json
import os

SRC_DIR = "exps/review/task_incremental_imr5t"
OUT_DIR = "exps/review/stream_smoke"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "hidelora", "treelora",
           "rainbowprompt", "progprompt"]

BUDGETS_MB = [250, 500]

# "Workable" roster (2026-07-19): excludes hidelora (confirmed total collapse --
# summation-free deployed forward can't recover training fragmented across many
# brief per-fold windows, see BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md) and
# progprompt (confirmed severe CIL-specific collapse, under active investigation
# as of this writing -- its CIL prompt-stack length tracks CHUNK count rather
# than real task count, diverging fast and degrading CIL sharply while TIL stays
# healthy). Both left out of new dataset smokes until further notice; task-
# splits/HP wiring for them is untouched and they can be re-added to METHODS
# once resolved.
WORKABLE_METHODS = ["seqlora", "olora", "inflora", "sketchlora", "treelora", "rainbowprompt"]

# (dataset, init_cls, increment, stop_after_tasks-for-smoke[, methods, budgets])
# `methods`/`budgets` default to the full METHODS roster / BUDGETS_MB list if omitted.
DATASET_CFG = {
    "cifar224": dict(init_cls=5, increment=5, smoke_tasks=5),
    "imagenetr": dict(init_cls=10, increment=10, smoke_tasks=5),
    # task splits per BOUNDARY_AGNOSTIC_IMPLEMENTATION_LOG.md's "Decisions made
    # autonomously" section: sun397 doesn't divide evenly into 20 tasks (17 +
    # 19*20 = 397), food101 likewise (6 + 19*5 = 101).
    "sun397": dict(init_cls=17, increment=20, smoke_tasks=5,
                   methods=WORKABLE_METHODS, budgets=[250]),
    "food101": dict(init_cls=6, increment=5, smoke_tasks=5,
                     methods=WORKABLE_METHODS, budgets=[250]),
}

# Optional per-dataset device pin (list[str] of GPU indices), applied to every
# method's generated config for that dataset. Leave a dataset out to keep
# whatever "device" the HP-source json already specifies.
DEVICE_OVERRIDE = {
    "imagenetr": ["4"],
}

os.makedirs(OUT_DIR, exist_ok=True)
for dataset, dcfg in DATASET_CFG.items():
    for budget_mb in dcfg.get("budgets", BUDGETS_MB):
        for m in dcfg.get("methods", METHODS):
            d = json.load(open(os.path.join(SRC_DIR, f"{m}.json")))
            d.pop("stop_after_tasks", None)
            d["dataset"] = dataset
            d["init_cls"] = dcfg["init_cls"]
            d["increment"] = dcfg["increment"]
            d["boundary_mode"] = "sample"
            d["stream_budget_mb"] = budget_mb
            d["stop_after_tasks"] = dcfg["smoke_tasks"]
            if dataset in DEVICE_OVERRIDE:
                d["device"] = DEVICE_OVERRIDE[dataset]
            d["prefix"] = f"stream_smoke_{dataset}_{budget_mb}mb_{m}"
            path = os.path.join(OUT_DIR, f"{dataset}_{budget_mb}mb_{m}.json")
            json.dump(d, open(path, "w"), indent=2)
            print("wrote", path)
