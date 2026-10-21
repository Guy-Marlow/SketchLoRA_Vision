"""Config generator for the olora_sketch_fd_divergence campaign (2026-10-21
user design): quantifies how SketchLoRA's and FD Sum's fixed-rank sketches
diverge from O-LoRA's unbounded-rank concatenated adapter over training, via
(1) spectral norm of the dense-update difference and (2) CKA between final
features, computed from per-boundary snapshots all three methods write under
this campaign's own run_logs tree.

3 methods (olora, sketchlora_align w/ merge_op=randsvd [SketchLoRA],
sketchlora_align w/ merge_op=freqdir_dense [FD Sum]) x 3 datasets
(cifar224-10t, imagenetr-20t, omnibenchmark1k-100t) x 3 seeds = 27 configs.
Every config shares: lora_rank=svd_rank=10, tuned_epoch=20, the project's
standing hyperparameters for each method (copied verbatim from the existing
main-table configs), differing only in {method, dataset, seed} plus the new
embed_drift_dir/adapter_snapshot_dir keys this campaign needs.

O-LoRA and SketchLoRA/FD-Sum MUST share the identical class_order per
(dataset, seed) for the comparison to be valid -- all three configs for a
given (dataset, seed) use shuffle=true with the SAME seed (not an explicit
class_order list), which utils/data_manager.py seeds identically
(np.random.seed(seed) -> np.random.permutation) regardless of model_name, so
this is already guaranteed without needing to pin an explicit class_order.
"""
import copy
import json
import os

CAMPAIGN = "olora_sketch_fd_divergence"
OUT_DIR = os.path.join("exps", CAMPAIGN)
SEEDS = (1993, 1996, 1999)
DATASETS = ("cifar224", "imagenetr", "omnibenchmark1k")

# (template path, extra overrides to strip to a clean base)
OLORA_TEMPLATES = {
    "cifar224": "exps/wave1_final_completion/olora_cifar224_s1993.json",
    "imagenetr": "exps/wave1_final_completion/olora_imagenetr_s1993.json",
    "omnibenchmark1k": "exps/wave1_final_completion/olora_omnibenchmark1k_s1993.json",
}
SKETCHLORA_TEMPLATES = {
    "cifar224": "exps/sketchlora_orth_wave1_datasets/cifar224_s1993.json",
    "imagenetr": "exps/sketchlora_orth_wave1_datasets/imagenetr_s1993.json",
    "omnibenchmark1k": "exps/sketchlora_orth_omnibenchmark1k/sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s1993.json",
}
# FD Sum has no existing Omnibenchmark-1k-100t config -- cifar224/imagenetr
# templates exist (freqdir_dense_wave1_v2); the omnibenchmark1k one is built
# from the SketchLoRA 100t template with merge_op swapped, matching exactly
# how freqdir_dense_wave1_v2's own configs relate to sketchlora_orth_wave1_
# datasets's (same hyperparams, only merge_op/prefix/diag_dir differ).
FDSUM_TEMPLATES = {
    "cifar224": "exps/freqdir_dense_wave1_v2/cifar224_s1993.json",
    "imagenetr": "exps/freqdir_dense_wave1_v2/imagenetr_s1993.json",
    "omnibenchmark1k": "exps/sketchlora_orth_omnibenchmark1k/sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s1993.json",
}

STRIP_KEYS = ("prefix", "sketchlora_diag_dir", "seed")


def base_config(path):
    cfg = json.load(open(path))
    for k in STRIP_KEYS:
        cfg.pop(k, None)
    return cfg


def write_config(cfg, method_tag, ds, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = [seed]
    prefix = "{}_{}_{}_s{}".format(CAMPAIGN, method_tag, ds, seed)
    cfg["prefix"] = prefix
    run_root = os.path.join("run_logs", CAMPAIGN, method_tag, ds)
    cfg["embed_drift_dir"] = os.path.join(run_root, "features", "s{}".format(seed)).replace("\\", "/")
    cfg["adapter_snapshot_dir"] = os.path.join(run_root, "snapshots", "s{}".format(seed)).replace("\\", "/")
    if "sketchlora_diag_dir" not in cfg and cfg.get("model_name") == "sketchlora_align":
        cfg["sketchlora_diag_dir"] = os.path.join(run_root, "diag", "s{}".format(seed)).replace("\\", "/")
    out_path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(method_tag, ds, seed))
    json.dump(cfg, open(out_path, "w"), indent=2)
    print("wrote", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ds in DATASETS:
        olora_base = base_config(OLORA_TEMPLATES[ds])
        sketch_base = base_config(SKETCHLORA_TEMPLATES[ds])
        fdsum_base = base_config(FDSUM_TEMPLATES[ds])
        # FD Sum's merge_op is forced to freqdir_dense regardless of which
        # template it came from -- the omnibenchmark1k template is SketchLoRA's
        # own (merge_op=randsvd) 100t config, reused verbatim except this swap.
        fdsum_base["merge_op"] = "freqdir_dense"
        assert sketch_base["merge_op"] == "randsvd", (ds, sketch_base["merge_op"])
        assert olora_base["model_name"] == "olora", (ds, olora_base["model_name"])
        for seed in SEEDS:
            write_config(olora_base, "olora", ds, seed)
            write_config(sketch_base, "sketchlora", ds, seed)
            write_config(fdsum_base, "fdsum", ds, seed)


if __name__ == "__main__":
    main()
