"""freqdir_dense merge_op campaign, wave 1 (2026-09-28 user request): FD
shrinkage on the randomized-SVD probe of the dense composite update (see
utils/freqdir_dense.py, models/sketchlora.py's merge_op="freqdir_dense"
branch) on 3 benchmarks x 3 seeds = 9 runs, to compare against the standing
SketchLoRA (orth, merge_op="randsvd") baseline -- same comparison
gen_freqdir_wave1_configs.py already ran for the ORIGINAL (decoupled-space)
merge_op="freqdir", extended to a third benchmark (OmniBenchmark-1k's
100-task split) that campaign didn't cover.

SOURCES: each config is loaded verbatim, per (dataset, seed), from the exact
baseline SketchLoRA config already used for the master-table numbers --
cifar224/imagenetr from exps/sketchlora_orth_wave1_datasets/<dataset>_
s<seed>.json, omnibenchmark1k from exps/sketchlora_orth_omnibenchmark1k/
sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s<seed>.json. Every
hyperparameter matches its source byte-for-byte except the three fields this
ablation actually changes: merge_op, sketchlora_diag_dir, and prefix.

Unlike gen_freqdir_wave1_configs.py, sketch_diag is left ON (True, inherited
from every source config) rather than force-disabled -- freqdir_dense DOES
form the dense delta_W (unlike the original "freqdir", which never does),
so the diagnostic block's recon_err/retained_energy measurements are
meaningful here, plus this merge_op's own freqdir_dense_delta/pre_shrink_
energy/post_shrink_energy fields (see models/sketchlora.py's _record_diag).
sketchlora_diag_dir is redirected to this campaign's own directory so it
can't collide with the source campaigns' own diag output for the same
(dataset, seed) cell.
"""
import json
import os

OUT_DIR = "exps/freqdir_dense_wave1"
RUN_LOGS_BASE = "run_logs/freqdir_dense_wave1"
SEEDS = [1993, 1996, 1999]

# (dataset tag used in this campaign's own filenames/prefixes) -> source config path template
SOURCES = {
    "cifar224": "exps/sketchlora_orth_wave1_datasets/cifar224_s{seed}.json",
    "imagenetr": "exps/sketchlora_orth_wave1_datasets/imagenetr_s{seed}.json",
    "omnibenchmark1k": "exps/sketchlora_orth_omnibenchmark1k/"
                        "sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s{seed}.json",
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
            cfg["prefix"] = "freqdir_dense_wave1_{}_s{}".format(dataset, seed)
            cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, dataset)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
