"""SketchLoRA QR-decomposition (merge_op="qrsvd") fill-out campaign
(2026-10-27 user request): mirrors every result this project already has
for the RandomSVD variant (merge_op="randsvd"), but with the QR-trick merge
(see scripts/benchmark_sketch_merge_algorithms.py's correctness check --
qrsvd is mathematically EXACT, not an approximation, so this campaign is
asking "does the production pipeline reproduce what the benchmark already
proved algebraically," not probing a new accuracy/compression tradeoff).

FOUR CATEGORIES, 3 seeds each (1993/1996/1999), model_name=sketchlora_align,
merge_op="qrsvd" throughout, every other hyperparameter identical to this
project's existing RandomSVD master-table/sensitivity configs (align_mode=
orth, svd_oversampling=10, lora_rank=10, lora_n_slots=2, sketch_diag=True,
final_metrics=True, print_forget=True -- same persistent-memory/runtime/
top-1%-accuracy logging the RandomSVD runs already have, nothing extra,
nothing missing):

1. main5 (15 runs): the 5 original master-table benchmarks -- CIFAR-100/
   cifar224-10t, ImageNet-R-20t, SUN397-10t, Food-101-20t, OmniBenchmark-
   1k-100t -- at the baseline cell (align_weight=0.5, svd_rank=10),
   mirroring exps/sketchlora_orth_wave1_datasets/*.json + exps/
   sketchlora_orth_omnibenchmark1k/*_fixedrank_orth05_*.json exactly,
   merge_op swapped to "qrsvd".

2. omni200t (3 runs): OmniBenchmark-1k's 200-task split (init_cls=5,
   increment=5), mirroring exps/omnibenchmark1k_200t/*.json exactly,
   merge_op swapped to "qrsvd". The single most expensive category here --
   200 sequential tasks x 3 seeds.

3. sensitivity_lambda (75 runs): sketchlora_align_weight in
   {0.1,0.3,0.5,0.7,0.9}, svd_rank fixed at 10, on all 5 datasets.

4. sensitivity_rank (75 runs): svd_rank in {5,10,15,20,25}, align_weight
   fixed at 0.5, on all 5 datasets.

Categories 3+4 are NOT deduplicated against each other or against category
1's own baseline cell (align_weight=0.5, svd_rank=10 appears once in each
of categories 1, 3, and 4) -- this is intentional, matching the user's own
explicit run-count arithmetic ("three seeds over five datasets for each
lambda value... and... each r_hat value... thirty runs over omnibenchmark-
1k-100t" = 5 lambda + 5 rank = 10 cells x 3 seeds = 30 for that one
dataset alone, which only adds up if the shared (0.5, 10) point is counted
twice, not deduped once). Total: 15 + 3 + 75 + 75 = 168 runs, with
OmniBenchmark-1k-100t contributing 3 (main5) + 30 (sensitivity) = 33 of
them and OmniBenchmark-1k-200t contributing 3 -- together the dominant
wall-time cost, exactly as the user flagged.

sketchlora_diag_dir is unique per category/dataset (standard convention
throughout this project) so no cell here, or from any earlier campaign,
can collide on a sketch_diag filename.
"""
import json
import os

OUT_BASE = "exps/sketchlora_qrsvd_fillout"
RUN_LOGS_BASE = "run_logs/sketchlora_qrsvd_fillout"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
    "omnibenchmark1k": dict(dataset="omnibenchmark1k", init_cls=10, increment=10),
}
OMNI200T = dict(dataset="omnibenchmark1k", init_cls=5, increment=5)

BASE = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora_align", backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    lora_merge=True, lora_train_merge=True,
    svd_oversampling=10, lora_n_slots=2, sketch_diag=True,
    sketchlora_lora_wd=0.0,
    sketchlora_align_mode="orth",
    sketchlora_align_weight=0.5,
    merge_op="qrsvd",
    device=["0"],
)


def write_cfg(category, ds_name, variant, seed, dataset_overrides, overrides, prefix_tag):
    cfg = dict(BASE)
    cfg.update(dataset_overrides)
    cfg["seed"] = [seed]
    cfg.update(overrides)
    cfg["prefix"] = "sketchlora_qrsvd_fillout_{}_s{}".format(prefix_tag, seed)
    cfg["sketchlora_diag_dir"] = "{}/{}/{}/diag".format(RUN_LOGS_BASE, category, ds_name)
    outdir = os.path.join(OUT_BASE, category, ds_name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "{}_s{}.json".format(variant, seed))
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def main():
    written = []

    # 1. main5
    for ds_name, ds_overrides in DATASETS.items():
        for seed in SEEDS:
            written.append(write_cfg(
                "main5", ds_name, ds_name, seed, ds_overrides,
                dict(svd_rank=10, sketchlora_align_weight=0.5),
                "main5_{}".format(ds_name)))

    # 2. omni200t
    for seed in SEEDS:
        written.append(write_cfg(
            "omni200t", "omnibenchmark1k_200t", "omnibenchmark1k_200t", seed, OMNI200T,
            dict(svd_rank=10, sketchlora_align_weight=0.5),
            "omni200t"))

    # 3. sensitivity_lambda
    for ds_name, ds_overrides in DATASETS.items():
        for w in (0.1, 0.3, 0.5, 0.7, 0.9):
            variant = "w{}".format(w)
            for seed in SEEDS:
                written.append(write_cfg(
                    "sensitivity_lambda", ds_name, variant, seed, ds_overrides,
                    dict(svd_rank=10, sketchlora_align_weight=w),
                    "sensitivity_lambda_{}_{}".format(ds_name, variant)))

    # 4. sensitivity_rank
    for ds_name, ds_overrides in DATASETS.items():
        for r in (5, 10, 15, 20, 25):
            variant = "r{}".format(r)
            for seed in SEEDS:
                written.append(write_cfg(
                    "sensitivity_rank", ds_name, variant, seed, ds_overrides,
                    dict(svd_rank=r, sketchlora_align_weight=0.5),
                    "sensitivity_rank_{}_{}".format(ds_name, variant)))

    print("wrote {} configs under {}/".format(len(written), OUT_BASE))


if __name__ == "__main__":
    main()
