"""Reads the final-task persistent_mb from a set of final_metrics JSON files
(expects exactly 3 -- one per seed, all status:"done"), prints the mean
rounded to the nearest whole MB to stdout. Used by
bankcap_wave1_imagenetr20t.slurm to turn SketchLoRA's 3-seed measured
footprint at a given rank cap into the bank_cap_mb budget for the other
methods. Fails loudly (nonzero exit, message on stderr) rather than
silently proceeding on a wrong file count or an unfinished run -- this
number becomes every other method's ACTUAL memory budget for the rest of
the campaign, so a silent wrong read here is not a safe failure mode.
"""

import argparse
import glob
import json
import statistics
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) != 3:
        print("FATAL: expected exactly 3 metrics files (one per seed), found {}: {}\n"
              "glob pattern: {}".format(len(files), files, args.glob), file=sys.stderr)
        sys.exit(1)

    vals = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "done":
            print("FATAL: {} is not status=done (found {!r}) -- run did not finish"
                  .format(path, d.get("status")), file=sys.stderr)
            sys.exit(1)
        vals.append(d["per_task"][-1]["persistent_mb"])

    mean = statistics.mean(vals)
    print("computed from: {} -> {}".format(files, vals), file=sys.stderr)
    print(round(mean))


if __name__ == "__main__":
    main()
