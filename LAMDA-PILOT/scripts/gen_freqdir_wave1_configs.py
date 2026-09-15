"""freqdir merge_op campaign, wave 1 (2026-09-15 user request): decoupled-space
Frequent-Directions SketchLoRA (see utils/freqdir.py, models/sketchlora.py's
merge_op="freqdir" branch) on 2 benchmarks x 3 seeds = 6 runs, to start
comparing against the standing SketchLoRA (orth, merge_op="randsvd") baseline.

SOURCES: each config is loaded verbatim, per seed, from the exact baseline
SketchLoRA config already used for the master-table numbers
(exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json) -- every
hyperparameter (epochs, lr, batch size, align_mode/weight, backbone, lora
rank, svd_rank) matches the existing baseline byte-for-byte except the three
fields this ablation actually changes: merge_op, sketch_diag (see below), and
prefix. This keeps the freqdir vs. randsvd comparison as close to
apples-to-apples as this project's other merge_op ablations have been.

sketch_diag explicitly set to False (not left to auto-disable-with-warning at
runtime) -- freqdir never forms the dense delta_W the diagnostic block reads,
per the user's own "hold off on that logging for now" (2026-09-15). Every
source config has sketch_diag=true (needed for the baseline's own randsvd
diagnostics); overridden off here rather than relying on the runtime warning
firing on every single run.

svd_oversampling is left in the config (harmless -- freqdir's construction
does exact SVDs and never reads self.oversampling) rather than stripped out,
so this config is otherwise a pure superset-compatible copy of the baseline.
"""
import json
import os

OUT_DIR = "exps/freqdir_wave1"
SOURCE_DIR = "exps/sketchlora_orth_wave1_datasets"
SEEDS = [1993, 1996, 1999]
DATASETS = ["cifar224", "imagenetr"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset in DATASETS:
        for seed in SEEDS:
            src_path = os.path.join(SOURCE_DIR, "{}_s{}.json".format(dataset, seed))
            cfg = json.load(open(src_path))
            cfg["merge_op"] = "freqdir"
            cfg["sketch_diag"] = False
            cfg["seed"] = [seed]
            cfg["prefix"] = "freqdir_wave1_{}_s{}".format(dataset, seed)
            cfg.pop("sketchlora_diag_dir", None)   # unused now that sketch_diag is off
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
