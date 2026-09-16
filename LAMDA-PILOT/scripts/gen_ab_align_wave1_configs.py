"""Independent A/B align-mode campaign, wave 1 (2026-09-16 user request):
vanilla (orth-default) SketchLoRA with four different sketch/residual
regularizer combinations, testing whether pulling adapters TOWARD shared
content (rather than orthogonalizing them apart) forces combined updates
toward lower rank. See models/sketchlora_align.py's module docstring for the
full mechanism (independent align_mode/align_b_mode, independent
align_weight/align_weight_b) -- verified via a 4-way smoke test (1993,
CIFAR-100, 3 tasks) before this campaign was built, including confirming the
weighted contribution of each active term lands in a comparable ~1-1.5
range regardless of which formula produced it.

Not related to freqdir (merge_op stays "randsvd", the project's standing
default) -- this varies ONLY the sketch/residual regularizer, everything
else (backbone, lr, epochs, svd_rank, merge_op) is vanilla orth SketchLoRA.

4 variants x 2 datasets x 3 seeds = 24 runs. Each config is loaded VERBATIM
from the exact baseline config already used for the master-table numbers
(exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json, merge_op=
"randsvd", align_mode="orth", align_weight=0.5 -- vanilla SketchLoRA)
and only the align-mode/weight keys + prefix are overridden:

  align_a_only    -- align_mode=align, align_weight=0.0025, align_b_mode=none
  align_b_only    -- align_mode=none,  align_weight=0.0025, align_b_mode=align
  align_ab        -- align_mode=align, align_weight=0.0025, align_b_mode=align
  orth_a_align_b  -- align_mode=orth,  align_weight=0.5, align_b_mode=align,
                     align_weight_b=0.0025  (A keeps vanilla SketchLoRA's own
                     scale; only B gets the smaller align-appropriate weight)

sketch_diag is left at the baseline's own value (True) -- unlike freqdir,
this campaign always forms the ordinary joint delta_W (merge_op="randsvd"
unchanged), so the diagnostic block works exactly as it does for every other
randsvd SketchLoRA run.
"""
import json
import os

OUT_DIR = "exps/ab_align_wave1"
SOURCE_DIR = "exps/sketchlora_orth_wave1_datasets"
SEEDS = [1993, 1996, 1999]
DATASETS = ["cifar224", "imagenetr"]

VARIANTS = {
    "align_a_only": {
        "sketchlora_align_mode": "align",
        "sketchlora_align_weight": 0.0025,
        "sketchlora_align_b_mode": "none",
    },
    "align_b_only": {
        "sketchlora_align_mode": "none",
        "sketchlora_align_weight": 0.0025,
        "sketchlora_align_b_mode": "align",
    },
    "align_ab": {
        "sketchlora_align_mode": "align",
        "sketchlora_align_weight": 0.0025,
        "sketchlora_align_b_mode": "align",
    },
    "orth_a_align_b": {
        "sketchlora_align_mode": "orth",
        "sketchlora_align_weight": 0.5,
        "sketchlora_align_b_mode": "align",
        "sketchlora_align_weight_b": 0.0025,
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset in DATASETS:
        for variant, overrides in VARIANTS.items():
            for seed in SEEDS:
                src_path = os.path.join(SOURCE_DIR, "{}_s{}.json".format(dataset, seed))
                cfg = json.load(open(src_path))
                cfg.update(overrides)
                cfg["seed"] = [seed]
                cfg["prefix"] = "ab_align_wave1_{}_{}_s{}".format(variant, dataset, seed)
                path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(variant, dataset, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
