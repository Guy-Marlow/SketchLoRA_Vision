"""Moves a just-finished run's CE2 JSON (written by trainer.py to the
hardcoded run_logs/ce2/<model_name>/ location -- no config field can
redirect it, same constraint as run_logs/final/<model_name>/metrics_*.json;
see _bankcap_run_done.py's own docstring for the parallel case) into this
campaign's own run_logs/<campaign>/ce2/<model_name>/ subfolder, so a single
`scp -r run_logs/<campaign>` pulls both the human-readable .out logs and the
machine-readable CE2 results together in one place (2026-08-31 user
request, after having to separately hunt for run_logs/ce2/*.json when
pulling the previous campaign's results locally).

Tag construction mirrors trainer.py's own `_tag2 = "{model}_{dataset}_
{prefix}_s{seed}"` / `CE2Logger(os.path.join("run_logs", "ce2", model_name),
_tag2, args)` exactly -- the same formula _bankcap_run_done.py already uses
for the parallel final_metrics path.

Usage: python scripts/_move_ce2_output.py <config.json> <dest_root>
  <dest_root> e.g. run_logs/ce_final_all/ce2 -- the model_name subfolder is
  created underneath it, matching run_logs/ce2/<model_name>'s own layout.

No-op (exit 0, no error) if ce2_enabled is false for this config, or if the
source file doesn't exist (e.g. the run crashed before CE2Logger ever wrote
anything) -- never raises, so it's safe to call unconditionally after every
run_one() attempt regardless of outcome.
"""
import json
import os
import shutil
import sys


def main():
    config_path = sys.argv[1]
    dest_root = sys.argv[2]
    with open(config_path) as f:
        cfg = json.load(f)

    if not cfg.get("ce2_enabled"):
        sys.exit(0)

    seed = cfg["seed"]
    seed = seed[0] if isinstance(seed, list) else seed
    prefix = cfg.get("prefix") or (cfg.get("boundary_mode") or "task")
    model_name = cfg["model_name"]
    tag = "{}_{}_{}_s{}".format(model_name, cfg["dataset"], prefix, seed)
    src = os.path.join("run_logs", "ce2", model_name, "ce2_{}.json".format(tag))

    if not os.path.exists(src):
        sys.exit(0)

    dest_dir = os.path.join(dest_root, model_name)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "ce2_{}.json".format(tag))
    shutil.move(src, dest)
    print("moved {} -> {}".format(src, dest))


if __name__ == "__main__":
    main()
