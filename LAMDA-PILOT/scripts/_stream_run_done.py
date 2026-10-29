"""Exit 0 iff the given streaming config's own run_logs/stream_... JSON
exists and shows partial=False (a clean finish) -- see models/stream_mixin.
py::_stream_checkpoint_write's own docstring: every mid-run checkpoint write
sets partial=True, and only trainer.py::_run_stream's own final write
(after stream_run() returns) overwrites it with partial=False. Mirrors
scripts/_bankcap_run_done.py's role for the final_metrics/oracle path, which
doesn't apply here since final_metrics is inert under boundary_mode="sample"
(trainer.py returns from _run_stream before that block is ever reached)."""
import json
import sys


def main(config_path):
    cfg = json.load(open(config_path))
    seed = cfg["seed"][0] if isinstance(cfg.get("seed"), (list, tuple)) else cfg.get("seed")
    out = "run_logs/stream_{}_{}_s{}.json".format(cfg["model_name"], cfg.get("prefix", "run"), seed)
    try:
        d = json.load(open(out))
    except (FileNotFoundError, json.JSONDecodeError):
        sys.exit(1)
    sys.exit(0 if d.get("partial") is False else 1)


if __name__ == "__main__":
    main(sys.argv[1])
