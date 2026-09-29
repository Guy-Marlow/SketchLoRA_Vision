"""freqdir_dense merge_op campaign, EXACT-SVD version (2026-09-28 user
request; algorithm fixed 2026-09-29) -- FD shrinkage on the dense composite
update's EXACT spectrum (see utils/freqdir_dense.py, models/sketchlora.py's
merge_op="freqdir_dense" branch) on 3 benchmarks x 3 seeds = 9 runs, to
compare against the standing SketchLoRA (orth, merge_op="randsvd") baseline
-- same comparison gen_freqdir_wave1_configs.py already ran for the ORIGINAL
(decoupled-space) merge_op="freqdir", extended to a third benchmark
(OmniBenchmark-1k's 100-task split) that campaign didn't cover.

DELIBERATELY A NEW CAMPAIGN NAME/PREFIX (freqdir_dense_wave1_exactsvd, not
freqdir_dense_wave1), not a same-name rerun -- the FIRST freqdir_dense_wave1
(randomized-probe version) already ran to completion and its results are
cached under run_logs/final/sketchlora_align/metrics_..._freqdir_dense_
wave1_<dataset>_s<seed>_s<seed>.json with status=="done". Reusing that same
prefix would make scripts/_bankcap_run_done.py's resumability check see
those (algorithmically WRONG -- shrinkage was verified to be a total no-op,
see utils/freqdir_dense.py's module docstring) results as already complete
and SILENTLY SKIP every run, never producing the corrected numbers. The new
prefix guarantees a clean slate; the old freqdir_dense_wave1 run/data is
left untouched as a documented negative result, not deleted.

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
energy/post_shrink_energy fields (see models/sketchlora.py's _record_diag) --
now a genuine, non-zero rent every fold after the first, unlike the original
randomized-probe version.
sketchlora_diag_dir is redirected to this campaign's own directory so it
can't collide with the source campaigns' (or the original freqdir_dense_
wave1's) own diag output for the same (dataset, seed) cell.
"""
import json
import os

OUT_DIR = "exps/freqdir_dense_wave1_exactsvd"
RUN_LOGS_BASE = "run_logs/freqdir_dense_wave1_exactsvd"
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
            cfg["prefix"] = "freqdir_dense_wave1_exactsvd_{}_s{}".format(dataset, seed)
            cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, dataset)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
