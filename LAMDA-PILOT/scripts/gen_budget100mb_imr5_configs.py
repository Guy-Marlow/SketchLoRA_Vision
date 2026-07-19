"""Generate budget-mode (100MB/chunk, 5 chunks) imagenet-r smoke configs for the
8-method budget-mode roster, reusing each method's already-finalized task-incremental
review config (exps/review/task_incremental_imr5t/*.json) as the HP source of truth --
only boundary_mode/budget_mb/stop_after_tasks change; init_cls/increment stay as the
nominal per-task class count (irrelevant to actual chunk boundaries under budget mode,
but still required by DataManager construction)."""
import json
import os

SRC_DIR = "exps/review/task_incremental_imr5t"
OUT_DIR = "exps/review/budget100mb_imr5"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "hidelora", "treelora",
           "rainbowprompt", "progprompt"]
DEVICE_CYCLE = ["0", "1", "4"]   # rotate across whichever GPUs are free at launch time

os.makedirs(OUT_DIR, exist_ok=True)
for i, m in enumerate(METHODS):
    d = json.load(open(os.path.join(SRC_DIR, f"{m}.json")))
    d.pop("stop_after_tasks", None)
    d["boundary_mode"] = "budget"
    d["budget_mb"] = 100
    d["stop_after_tasks"] = 5
    d["device"] = [DEVICE_CYCLE[i % len(DEVICE_CYCLE)]]
    d["prefix"] = f"budget100mb_imr5_{m}"
    path = os.path.join(OUT_DIR, f"{m}.json")
    json.dump(d, open(path, "w"), indent=2)
    print("wrote", path)
