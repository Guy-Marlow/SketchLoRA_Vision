"""Generate budget-mode (250MB/chunk, 8 chunks out of 97 total) omnibenchmark-1k
smoke configs for the 8-method budget-mode roster, reusing each method's
already-finalized task-incremental review config as the HP source of truth.
InfLoRA gets the fixed lamb=lame=0.975 budget-mode convention (not the annealed
0.95/1.0 used for task-incremental)."""
import json
import os

SRC_DIR = "exps/review/task_incremental_imr5t"
OUT_DIR = "exps/review/budget250mb_omni8"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "hidelora", "treelora",
           "rainbowprompt", "progprompt"]

os.makedirs(OUT_DIR, exist_ok=True)
for m in METHODS:
    d = json.load(open(os.path.join(SRC_DIR, f"{m}.json")))
    d.pop("stop_after_tasks", None)
    d["dataset"] = "omnibenchmark1k"
    d["boundary_mode"] = "budget"
    d["budget_mb"] = 250
    d["stop_after_tasks"] = 8
    if m == "inflora":
        d["lamb"] = 0.975
        d["lame"] = 0.975
    d["prefix"] = f"budget250mb_omni8_{m}"
    path = os.path.join(OUT_DIR, f"{m}.json")
    json.dump(d, open(path, "w"), indent=2)
    print("wrote", path)
