"""Exit 0 iff the given streaming config's own run_logs/{tag}_... JSON
exists and shows partial=False (a clean finish) -- see models/stream_mixin.
py::_stream_checkpoint_write's own docstring: every mid-run checkpoint write
sets partial=True, and only trainer.py::_run_stream's own final write
(after the driver method returns) overwrites it with partial=False. Mirrors
scripts/_bankcap_run_done.py's role for the final_metrics/oracle path, which
doesn't apply here since final_metrics is inert under boundary_mode="sample"
(trainer.py returns from _run_stream before that block is ever reached).

2026-11-02: optional second CLI arg (tag, default "stream") added so this
same checker works for boundary_mode="loss_plateau"/"spectral_plateau" too --
trainer.py::_run_stream's own `tag` param changes the checkpoint filename
(run_logs/{tag}_{model_name}_{prefix}_s{seed}.json), "stream" only matches
boundary_mode="sample"'s own files. Every existing call site (no second arg)
is unaffected -- defaults to the original "stream" behavior unchanged."""
import json
import sys


def main(config_path, tag="stream"):
    cfg = json.load(open(config_path))
    seed = cfg["seed"][0] if isinstance(cfg.get("seed"), (list, tuple)) else cfg.get("seed")
    out = "run_logs/{}_{}_{}_s{}.json".format(tag, cfg["model_name"], cfg.get("prefix", "run"), seed)
    try:
        d = json.load(open(out))
    except (FileNotFoundError, json.JSONDecodeError):
        sys.exit(1)
    sys.exit(0 if d.get("partial") is False else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "stream")
