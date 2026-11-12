"""Exit 0 iff the batch-incremental output JSON for the given config exists
and shows partial==False; exit 1 otherwise (missing, or partial==True).

NOT the same check as scripts/_bankcap_run_done.py -- boundary_mode==
"kl_batch_inc" routes through trainer.py::_run_stream, which writes a
DIFFERENT output schema and path (run_logs/kl_batch_inc_<model_name>_
<prefix>_s<seed>.json, keys {args_subset, results, partial}) with no
run_logs/final/<model>/metrics_*.json / status=="done" MetricsLogger
output at all. _run_stream also writes this file exactly ONCE, after the
full task loop completes -- there is no mid-run checkpoint write anywhere
in models/sketchlora_batchinc.py::kl_batch_inc_run, so a run killed before
this file exists has made no resumable progress whatsoever and will
restart from task 0 on the next submission, not resume partway through.
"""
import json
import os
import sys


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    seed = cfg["seed"]
    seed = seed[0] if isinstance(seed, list) else seed
    prefix = cfg.get("prefix") or "run"
    out_path = os.path.join("run_logs", "kl_batch_inc_{}_{}_s{}.json".format(
        cfg["model_name"], prefix, seed))

    if not os.path.exists(out_path):
        sys.exit(1)
    with open(out_path) as f:
        d = json.load(f)
    sys.exit(0 if d.get("partial") is False else 1)


if __name__ == "__main__":
    main()
