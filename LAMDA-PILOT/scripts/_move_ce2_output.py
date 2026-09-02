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

Usage: python scripts/_move_ce2_output.py <config.json> <dest_root> [dest_name]
  <dest_root> e.g. run_logs/ce_final_all/ce2 -- the destination subfolder is
  created underneath it, matching run_logs/ce2/<model_name>'s own layout.
  [dest_name] optional override for that subfolder's name (and the copied
  file's own model-portion of its filename) -- defaults to the config's
  real model_name if omitted, matching every existing caller unchanged.

  Use the override when a config's real model_name is an implementation
  detail the project has decided to REPORT under a different name (2026-09-01
  case: exps/sketchlora_fixedrank_orth05_imagenetr_s1993.json's model_name is
  "sketchlora_align" -- required, that's the actual Learner class the orth
  mechanism lives in, see utils/factory.py's dispatch -- but "SketchLoRA" is
  now this project's standing name for that config specifically, per user
  convention, so its CE2 output should land as/overwrite ce2/sketchlora/,
  not create a separate ce2/sketchlora_align/). The SOURCE path this script
  reads from is still keyed by the config's real model_name regardless
  (that part of the location is hardcoded by trainer.py, not overridable) --
  only the DESTINATION is renamed.

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
    dest_name_override = sys.argv[3] if len(sys.argv) > 3 else None
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

    dest_name = dest_name_override or model_name
    dest_tag = tag if dest_name == model_name else "{}_{}_{}_s{}".format(
        dest_name, cfg["dataset"], prefix, seed)
    dest_dir = os.path.join(dest_root, dest_name)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "ce2_{}.json".format(dest_tag))
    shutil.move(src, dest)
    print("moved {} -> {}".format(src, dest))


if __name__ == "__main__":
    main()
