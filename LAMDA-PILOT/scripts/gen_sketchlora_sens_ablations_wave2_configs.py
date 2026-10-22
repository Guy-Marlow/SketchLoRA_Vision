"""SketchLoRA sensitivity + ablation campaign, wave 2 (2026-10-22 user
request): extends sketchlora_orth_sens_ablations (CIFAR-100/ImageNet-R only)
and sketchlora_orth_sens_omnibenchmark1k (OmniBenchmark-1k rank/weight grid
only) to finish the remaining cells across SUN397, Food-101, and
OmniBenchmark-1k's 100-task split, plus a NEW align-mode ablation axis
neither earlier campaign covered.

FOUR AXES, 3 seeds each (1993/1996/1999), model_name=sketchlora_align
throughout:

1. ORTH-WEIGHT SENSITIVITY: sketchlora_align_weight in {0.1,0.3,0.5,0.7,0.9},
   svd_rank fixed at 10, on SUN397 and Food-101.
2. SKETCH-RANK SENSITIVITY: svd_rank in {5,10,15,20,25}, align_weight fixed
   at 0.5, on SUN397 and Food-101.
   Axes 1+2 share the (weight=0.5, rank=10) baseline cell -- a single
   9-point grid per dataset, not 10 independent runs (same convention as
   scripts/gen_sketchlora_orth_sens_ablations_configs.py). That cell is
   already the master-table SketchLoRA result for each dataset
   (exps/sketchlora_orth_wave1_datasets/{sun397,food101}_s<seed>.json) and is
   REUSED verbatim, not regenerated or rerun.
3. MERGE-OP COMPARISON: merge_op in {exactsvd, randsvd+random-rank-selection
   ("Random Select"), freqdir ("FD Separate"), freqdir_dense ("FD Sum"),
   countsketch}, each at the baseline regularizer (align_mode=orth,
   align_weight=0.5, svd_rank=10) -- only the compression/merge algorithm
   varies -- on OmniBenchmark-1k, SUN397, and Food-101. (The implicit 6th
   point, plain randsvd, is the master-table baseline itself and is not
   regenerated here.)
4. ALIGN-MODE ABLATION: three regularizer variants relative to the vanilla
   orth baseline -- without_orth (no regularizer at all: align_mode=none,
   align_b_mode=none), aligned_a (align_mode=align, weight=0.0025,
   align_b_mode=none), orth_a_aligned_b (align_mode=orth, weight=0.5,
   align_b_mode=align, weight_b=0.0025) -- on OmniBenchmark-1k, SUN397, and
   Food-101. Same two of the four variants scripts/gen_ab_align_wave1_configs
   .py already ran (align_a_only/orth_a_align_b; align_b_only/align_ab were
   not requested here), plus a genuinely new "without_orth" cell neither
   prior campaign had on any dataset.

Run counts: axes 1+2 = 9 cells x 2 datasets x 3 seeds, minus 1 reused cell x
2 datasets x 3 seeds = 48 new runs. Axis 3 = 5 variants x 3 datasets x 3
seeds = 45 new runs. Axis 4 = 3 variants x 3 datasets x 3 seeds = 27 new
runs. Total 120 new runs (plus 6 reused configs that are not re-executed).

sketch_diag=True + a per-category/per-dataset sketchlora_diag_dir (same
collision-avoidance convention as every other sens/ablation generator in
this project) so no two cells here, or cells from an earlier campaign, can
ever write to the same sketch_diag filename.
"""
import json
import os

OUT_BASE = "exps/sketchlora_sens_ablations_wave2"
RUN_LOGS_BASE = "run_logs/sketchlora_sens_ablations_wave2"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
    "omnibenchmark1k": dict(dataset="omnibenchmark1k", init_cls=10, increment=10),
}

SENSITIVITY_DATASETS = ["sun397", "food101"]
MERGEOP_DATASETS = ["omnibenchmark1k", "sun397", "food101"]
ALIGN_ABLATION_DATASETS = ["omnibenchmark1k", "sun397", "food101"]

# Baseline (weight=0.5, rank=10) master-table SketchLoRA configs -- reused
# verbatim for the sensitivity grid's shared cell. Not regenerated/rerun.
BASELINE_SOURCE_DIR = "exps/sketchlora_orth_wave1_datasets"

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
    merge_op="randsvd",
    device=["0"],
)


def write_cfg(category, ds_name, variant, seed, overrides):
    cfg = dict(BASE)
    cfg.update(DATASETS[ds_name])
    cfg["seed"] = [seed]
    cfg.update(overrides)
    cfg["prefix"] = "sketchlora_sens_ablations_wave2_{}_{}_{}_s{}".format(
        category, ds_name, variant, seed)
    # forward slashes explicit (not os.path.join) -- built on Windows, read
    # back by main.py on the Linux cluster (same convention as every other
    # sens/ablation generator in this project).
    cfg["sketchlora_diag_dir"] = "{}/{}/{}/diag".format(RUN_LOGS_BASE, category, ds_name)
    outdir = os.path.join(OUT_BASE, category, ds_name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "{}_s{}.json".format(variant, seed))
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def sensitivity_cells():
    """9-point (weight, rank) grid, deduplicated -- see module docstring."""
    cells = {}
    for w in (0.1, 0.3, 0.5, 0.7, 0.9):
        cells[(w, 10)] = "w{}_r10".format(w)
    for r in (5, 10, 15, 20, 25):
        cells[(0.5, r)] = "w0.5_r{}".format(r)
    return cells


MERGE_OPS = {
    "exactsvd": dict(merge_op="exactsvd"),
    "randomsel": dict(merge_op="randsvd", sketchlora_rank_selection="random"),
    "fdseparate": dict(merge_op="freqdir"),
    "fdsum": dict(merge_op="freqdir_dense"),
    "countsketch": dict(merge_op="countsketch"),
}

ALIGN_VARIANTS = {
    "without_orth": dict(
        sketchlora_align_mode="none", sketchlora_align_weight=0.0,
        sketchlora_align_b_mode="none"),
    "aligned_a": dict(
        sketchlora_align_mode="align", sketchlora_align_weight=0.0025,
        sketchlora_align_b_mode="none"),
    "orth_a_aligned_b": dict(
        sketchlora_align_mode="orth", sketchlora_align_weight=0.5,
        sketchlora_align_b_mode="align", sketchlora_align_weight_b=0.0025),
}


def main():
    written = []
    reused = []

    for ds_name in SENSITIVITY_DATASETS:
        for (weight, rank), variant in sensitivity_cells().items():
            if (weight, rank) == (0.5, 10):
                for seed in SEEDS:
                    reused.append((ds_name, variant, seed, os.path.join(
                        BASELINE_SOURCE_DIR, "{}_s{}.json".format(ds_name, seed))))
                continue
            for seed in SEEDS:
                written.append(write_cfg("sensitivity", ds_name, variant, seed, dict(
                    svd_rank=rank, sketchlora_align_weight=weight,
                )))

    for ds_name in MERGEOP_DATASETS:
        for variant, overrides in MERGE_OPS.items():
            for seed in SEEDS:
                cfg_overrides = dict(svd_rank=10, sketchlora_align_weight=0.5)
                cfg_overrides.update(overrides)
                written.append(write_cfg("mergeop", ds_name, variant, seed, cfg_overrides))

    for ds_name in ALIGN_ABLATION_DATASETS:
        for variant, overrides in ALIGN_VARIANTS.items():
            for seed in SEEDS:
                cfg_overrides = dict(svd_rank=10)
                cfg_overrides.update(overrides)
                written.append(write_cfg("align_ablation", ds_name, variant, seed, cfg_overrides))

    print("wrote {} configs under {}/".format(len(written), OUT_BASE))
    print("reused {} existing baseline configs (not regenerated/rerun):".format(len(reused)))
    for ds_name, variant, seed, path in reused:
        print("  {} / {} / s{} -> {}".format(ds_name, variant, seed, path))


if __name__ == "__main__":
    main()
