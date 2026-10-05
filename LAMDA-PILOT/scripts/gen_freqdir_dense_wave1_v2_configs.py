"""freqdir_dense merge_op campaign, CURRENT (randomized-probe, window-
restricted) implementation -- 2026-09-30 user request: run locally on this
A5000 for ImageNet-R-20t and CIFAR-100-10t, 3 seeds each = 6 runs.

This is deliberately a NEW campaign name/prefix (freqdir_dense_wave1_v2, not
freqdir_dense_wave1 or freqdir_dense_wave1_exactsvd):
  - freqdir_dense_wave1 (no suffix) is the ORIGINAL run under the FIRST
    freqdir_dense implementation, which was verified to be a total no-op
    (shrinkage delta ~1e-16 to 1e-17 every fold -- bit-identical to plain
    randsvd, because it shrunk the randomized probe's oversampling-padding
    singular values instead of the real spectrum). Its cached results are a
    documented negative result, not a valid baseline to resume into.
  - freqdir_dense_wave1_exactsvd is a DIFFERENT algorithm (exact torch.linalg.
    svd on the dense composite update, O(dim^3) per fold) that briefly lived
    at commit beb04c8 before being superseded by the current fix. Its name
    would be actively misleading for this campaign, which uses HEAD's
    implementation (randomized SVD probe, shrinkage restricted to
    S[:composite_rank] to exclude the oversampling-padding noise floor --
    see models/sketchlora.py's merge_op=="freqdir_dense" branch and
    utils/freqdir_dense.py, commit 1a6ae6f).

Reusing either old prefix would make scripts/_bankcap_run_done.py's
resumability check see stale results and silently skip these runs.

SOURCES: each config loaded verbatim, per (dataset, seed), from the exact
baseline SketchLoRA config used for the master-table numbers
(exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json). Every
hyperparameter matches its source byte-for-byte except merge_op,
sketchlora_diag_dir, and prefix.
"""
import json
import os

OUT_DIR = "exps/freqdir_dense_wave1_v2"
RUN_LOGS_BASE = "run_logs/freqdir_dense_wave1_v2"
SEEDS = [1993, 1996, 1999]

SOURCES = {
    "cifar224": "exps/sketchlora_orth_wave1_datasets/cifar224_s{seed}.json",
    "imagenetr": "exps/sketchlora_orth_wave1_datasets/imagenetr_s{seed}.json",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset, src_template in SOURCES.items():
        for seed in SEEDS:
            src_path = src_template.format(seed=seed)
            cfg = json.load(open(src_path))
            cfg["merge_op"] = "freqdir_dense"
            cfg["seed"] = [seed]
            cfg["prefix"] = "freqdir_dense_wave1_v2_{}_s{}".format(dataset, seed)
            cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, dataset)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
