import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ImageNet-R, 3-task smoke (seed 1993), settled per-method configs, GPU2 sequential.
# HiDeLoRA excluded: its deployed forward selects a single stored slot rather than
# summing frozen slots (unlike O-LoRA/InfLoRA/TreeLoRA), so it collapses under CIL
# eval without oracle task-id -- not a comparable data point here.
data = [
    ("SeqLoRA",       35.29),
    ("SketchLoRA",    35.47),
    ("O-LoRA",        40.87),
    ("InfLoRA",       40.87),
    ("TreeLoRA",      40.87),
    ("RainbowPrompt", 70.34),
    ("EASE",          141.72),
    ("TUNA",          141.02),
]
df = pd.DataFrame(data, columns=["Method", "gflops"])

sns.set_theme(style="whitegrid", context="talk")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x="Method", y="gflops", hue="Method", palette="pastel", legend=False, ax=ax)
ax.set_title("Per-task inference cost (ImageNet-R)")
ax.set_ylabel("GFLOPs / image")
ax.set_xlabel("")
ax.set_ylim(0, 150)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "gflops_pastel.png")
fig.savefig(out_path, dpi=150)
print("saved", out_path)
