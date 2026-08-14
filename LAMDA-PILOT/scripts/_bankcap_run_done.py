"""Exit 0 iff the final_metrics JSON for the given config shows
status=="done"; exit 1 otherwise (file missing, run in progress, or
final_metrics not enabled in this config). Used by
bankcap_wave1_imagenetr20t.slurm's run_one() as the completion check.

NOT a check of the .out log's "Average Accuracy (CNN):" line -- trainer.py
prints that once per TASK (every checkpoint), not once per completed run
(mlog.finalize(), which sets status=="done", runs once, after the whole
task loop, strictly later). Grepping the log for it would falsely mark a
run killed by a wall-time cutoff after its first task as "done" on
resubmission -- run_one would skip it forever while its metrics JSON stays
stuck below status=="done" forever, exactly the signal
bankcap_mean_from_metrics.py depends on to compute the derived cap. Using
the SAME status=="done" signal in both places means the two can never
disagree about whether a run is actually finished.

Tag construction mirrors trainer.py's own `_tag = "{model}_{dataset}_
{prefix}_s{seed}"` / `MetricsLogger(os.path.join("run_logs","final",
model_name), _tag, args)` exactly -- see trainer.py's `_train()`.
"""

import json
import os
import sys


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    if not cfg.get("final_metrics"):
        sys.exit(1)

    seed = cfg["seed"]
    seed = seed[0] if isinstance(seed, list) else seed
    prefix = cfg.get("prefix") or (cfg.get("boundary_mode") or "task")
    tag = "{}_{}_{}_s{}".format(cfg["model_name"], cfg["dataset"], prefix, seed)
    metrics_path = os.path.join("run_logs", "final", cfg["model_name"], "metrics_{}.json".format(tag))

    if not os.path.exists(metrics_path):
        sys.exit(1)
    with open(metrics_path) as f:
        d = json.load(f)
    sys.exit(0 if d.get("status") == "done" else 1)


if __name__ == "__main__":
    main()
