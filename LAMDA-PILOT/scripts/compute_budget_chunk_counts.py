"""Compute the total number of budget-mode chunks (intervals) for a grid of
(dataset, budget_mb) combinations, using the real BudgetStreamManager -- no
training, just data loading + chunk construction. Reports the degenerate-chunk
RuntimeError (zero-new-class chunk) per-combination if one is hit, rather than
silently skipping it.
"""
import sys
sys.path.insert(0, ".")
from utils.data_manager import DataManager
from utils.budget_stream import BudgetStreamManager

DATASETS = ["cifar224", "imagenetr", "omnibenchmark1k", "sun397", "food101"]
BUDGETS = [100, 150, 200, 250, 300, 400, 500]

results = {}
for dataset in DATASETS:
    args = {"model_name": "olora", "shuffle": True}
    dm = DataManager(dataset, True, 1993, 10, 10, args)
    for budget in BUDGETS:
        key = (dataset, budget)
        try:
            bsm = BudgetStreamManager(dm, budget_mb=budget, seed=1993)
            results[key] = ("OK", bsm.nb_tasks)
        except RuntimeError as e:
            msg = str(e)
            # pull out the zero-chunk indices for a compact report
            results[key] = ("DEGENERATE", msg.split("indices ")[1].split(")")[0] if "indices" in msg else msg[:80])

print(f"{'dataset':<18}{'budget_mb':<12}{'status':<12}{'nb_tasks / detail'}")
for dataset in DATASETS:
    for budget in BUDGETS:
        status, detail = results[(dataset, budget)]
        print(f"{dataset:<18}{budget:<12}{status:<12}{detail}")
