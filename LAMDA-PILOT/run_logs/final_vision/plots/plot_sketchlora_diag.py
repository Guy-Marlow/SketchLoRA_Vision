"""SketchLoRA-specific adapter diagnostics for the final_vision H200 batch.
Source: run_logs/final_vision/run_logs/sketchlora_diag_adapt0.01_ball_<tag>_seed<seed>.json
(written by models/sketchlora.py::_record_diag), NOT the metrics_*.json files --
this is a separate artifact tracking the compressed sketch's singular spectrum at
every compression event. tag = ic{init_cls}i{increment}, uniquely identifying each
dataset under this batch's (old) split convention: cifar224=ic5i5, food101=ic6i5,
sun397=ic17i20. All three datasets have all 3 seeds (1993/1994/1995) here -- SD
bands throughout.

Metrics plotted (per task, mean across the 24 q/v modules unless noted):
  - r_hat_mean:      mean compressed sketch rank across modules
  - r_hat_total:     summed rank across all 24 modules (proportional to adapter size)
  - retained_mean:   mean fraction of ||delta_W||^2 energy retained by the truncation
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DIAG_DIR = "run_logs/final_vision/run_logs"
OUT = "run_logs/final_vision/plots"
DATASET_TAGS = {"cifar224": "ic5i5", "food101": "ic6i5", "sun397": "ic17i20"}
SEEDS = [1993, 1994, 1995]
METRICS = ["r_hat_mean", "r_hat_total", "retained_mean"]
METRIC_LABELS = {
    "r_hat_mean": "mean rank (r̂/module)",
    "r_hat_total": "total rank (summed, 24 modules)",
    "retained_mean": "retained energy fraction",
}

os.makedirs(OUT, exist_ok=True)

rows = []
for dataset, tag in DATASET_TAGS.items():
    for seed in SEEDS:
        path = f"{DIAG_DIR}/sketchlora_diag_adapt0.01_ball_{tag}_seed{seed}.json"
        if not os.path.exists(path):
            print("SKIP (missing):", path)
            continue
        records = json.load(open(path))
        for rec in records:
            for metric in METRICS:
                val = rec.get(metric)
                if val is None:
                    continue
                rows.append(dict(dataset=dataset, seed=seed, task=rec["task"],
                                 metric=metric, value=val))

df = pd.DataFrame(rows)
sns.set_theme(style="whitegrid")

for dataset in DATASET_TAGS:
    for metric in METRICS:
        sub = df[(df.dataset == dataset) & (df.metric == metric)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.lineplot(data=sub, x="task", y="value", marker="o", ax=ax,
                    color="#a13d63", errorbar="sd")
        n_seeds = sub.seed.nunique()
        ax.set_title(f"{dataset} — SketchLoRA {METRIC_LABELS[metric]} (n={n_seeds}, ±1 SD)",
                    fontsize=11.5)
        ax.set_xlabel("task")
        ax.set_ylabel(METRIC_LABELS[metric])
        fig.tight_layout()
        out = f"{OUT}/{dataset}_sketchlora_{metric}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out)

# -- cross-dataset comparison: one plot per metric, all 3 datasets overlaid --
DATASET_LABELS = {"cifar224": "CIFAR-100", "food101": "Food101", "sun397": "SUN397"}
CROSS_METRICS = ["r_hat_mean", "retained_mean"]
for metric in CROSS_METRICS:
    sub = df[df.metric == metric].copy()
    if sub.empty:
        continue
    sub["Dataset"] = sub.dataset.map(DATASET_LABELS)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=sub, x="task", y="value", hue="Dataset", marker="o", ax=ax,
                hue_order=["CIFAR-100", "Food101", "SUN397"], errorbar="sd")
    ax.set_title(f"SketchLoRA {METRIC_LABELS[metric]} across datasets (n=3 seeds each, ±1 SD)",
                fontsize=11.5)
    ax.set_xlabel("task")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(title=None, frameon=False)
    fig.tight_layout()
    out = f"{OUT}/sketchlora_{metric}_all_datasets.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)
