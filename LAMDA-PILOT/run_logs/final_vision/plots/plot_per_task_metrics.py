"""Per-task metric comparison plots for the final_vision H200 batch. One graph per
(dataset, metric) pairing, one curve per method present for that dataset. Bands are
standard deviation across seeds (not the seaborn-default bootstrap CI). Methods with
fewer seeds/tasks than the rest of a given plot are auto-flagged via the "partial"
annotation rather than silently averaged in or excluded.

Re-rebuilt 2026-07-23 after another cluster run_logs/ refresh -- TreeLoRA now has
real data: cifar224 seeds 1993/1994 done (1995 still running, partial), food101
seed 1994 running (partial), no sun397 data yet (will simply be absent there).
"""
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ROOT = "run_logs/final_vision/run_logs/final"
OUT = "run_logs/final_vision/plots"
METHODS = ["seqlora", "olora", "inflora", "sketchlora", "treelora"]
METRICS = ["cil_top1", "train_seconds", "eval_seconds", "persistent_mb", "peak_vram_mb"]
METRIC_LABELS = {
    "cil_top1": "CIL top-1 (%)",
    "train_seconds": "train time (s/task)",
    "eval_seconds": "eval time (s/task)",
    "persistent_mb": "persistent state (MB)",
    "peak_vram_mb": "peak VRAM (MB)",
}
DATASET_SEEDS = {
    "cifar224": [1993, 1994, 1995],
    "food101": [1993, 1994],
    "sun397": [1993],
}
METHOD_LABELS = {"seqlora": "SeqLoRA", "olora": "O-LoRA", "inflora": "InfLoRA",
                 "sketchlora": "SketchLoRA", "treelora": "TreeLoRA"}
HUE_ORDER = ["SeqLoRA", "O-LoRA", "InfLoRA", "SketchLoRA", "TreeLoRA"]

os.makedirs(OUT, exist_ok=True)


def load(method, dataset, seed):
    pattern = f"{ROOT}/{method}/metrics_{method}_{dataset}_final_vision_{dataset}_{method}_s{seed}_s{seed}.json"
    matches = glob.glob(pattern)
    if not matches:
        return None
    d = json.load(open(matches[0]))
    if d.get("status") not in ("done", "running"):
        return None
    return d


rows = []
for dataset, seeds in DATASET_SEEDS.items():
    for method in METHODS:
        for seed in seeds:
            d = load(method, dataset, seed)
            if d is None:
                print(f"SKIP (missing): {method} {dataset} seed{seed}")
                continue
            n = len(d["per_task"])
            if n < 20:
                print(f"NOTE (partial, {n}/20 tasks): {method} {dataset} seed{seed}")
            for t in d["per_task"]:
                for metric in METRICS:
                    val = t.get(metric)
                    if val is None:
                        continue
                    rows.append(dict(dataset=dataset, method=method, seed=seed,
                                     task=t["task"], metric=metric, value=val))

df = pd.DataFrame(rows)
sns.set_theme(style="whitegrid")

for dataset in DATASET_SEEDS:
    for metric in METRICS:
        sub = df[(df.dataset == dataset) & (df.metric == metric)]
        if sub.empty:
            continue
        sub = sub.assign(Method=sub.method.map(METHOD_LABELS))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        present = [m for m in HUE_ORDER if m in sub.Method.unique()]
        sns.lineplot(data=sub, x="task", y="value", hue="Method", marker="o", ax=ax,
                    hue_order=present, errorbar="sd")
        n_seeds = sub.seed.nunique()
        seed_note = f"n={n_seeds} seeds, +/-1 SD" if n_seeds > 1 else f"seed {sub.seed.iloc[0]}"
        ax.set_title(f"{dataset} — {METRIC_LABELS[metric]} ({seed_note})")
        ax.set_xlabel("task")
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric == "cil_top1":
            ax.set_ylim(0, 100)

        # Per-(method,task) seed coverage -- catches a seed that drops out partway
        # (e.g. TreeLoRA's still-running seed dropping from 3 seeds' worth of band
        # to 2 partway through), not just a method whose ENTIRE seed set lags.
        coverage = sub.groupby(["Method", "task"])["seed"].nunique()
        max_task_by_method = sub.groupby("Method")["task"].max()
        n_seeds_by_method = sub.groupby("Method")["seed"].nunique()
        overall_max_task = max_task_by_method.max()
        overall_max_seeds = n_seeds_by_method.max()
        partial = []
        for m in max_task_by_method.index:
            min_cov = coverage[m].min()
            if max_task_by_method[m] < overall_max_task or min_cov < overall_max_seeds:
                partial.append(
                    f"{m} (through task {max_task_by_method[m]}, "
                    f"{min_cov}-{n_seeds_by_method[m]} seeds depending on task)"
                    if min_cov < n_seeds_by_method[m] else
                    f"{m} (through task {max_task_by_method[m]}, {n_seeds_by_method[m]} seed"
                    f"{'s' if n_seeds_by_method[m] > 1 else ''})"
                )
        if partial:
            ax.text(0.02, 0.02, "partial: " + "; ".join(partial), transform=ax.transAxes,
                    fontsize=7.5, color="0.4", va="bottom", ha="left")

        ax.legend(title=None, frameon=False)
        fig.tight_layout()
        out = f"{OUT}/{dataset}_{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out)
