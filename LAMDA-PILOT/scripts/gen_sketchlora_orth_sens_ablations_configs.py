"""SketchLoRA orth sensitivity + ablation campaign (2026-09-02 user request).
Two datasets (ImageNet-R-20t, CIFAR-100/cifar224-10t), 3 seeds each
(1993/1996/1999), model_name=sketchlora_align throughout (the orth variant is
the standing "SketchLoRA" default -- see models/sketchlora_align.py). Every
config: lora_rank=10, batch=48, tuned_epoch=20, merge_op=randsvd unless an
ablation overrides it, sketchlora_align_mode="orth".

TWO SENSITIVITY AXES, sharing one baseline cell (svd_rank=10,
align_weight=0.5) so they're a single 9-point grid, not 10 separate runs:
  - orth strength: sketchlora_align_weight in {0.1, 0.3, 0.5, 0.7, 0.9},
    svd_rank fixed at 10.
  - sketch rank: svd_rank in {5, 10, 15, 20, 25}, align_weight fixed at 0.5.
No new mechanism needed for the rank axis -- fixed svd_rank=R truncation
IS "concatenate losslessly while true rank <= R, start truncating for real
once it exceeds R" by construction (randsvd_probe's spectrum beyond the true
composite rank is ~0-magnitude numerical padding, so slicing to R when
R >= true rank changes nothing material; slicing to R<true rank is the
genuine truncation). Confirmed against the actual _compress code before
writing this generator, not assumed.

BASELINE-CELL REUSE (2026-09-02 user decision): (svd_rank=10,
align_weight=0.5) on imagenetr is NOT regenerated here -- it's byte-identical
to the already-completed exps/sketchlora_fixedrank_orth05_imagenetr_s1993.json
(seed 1993 -- 1996/1999 also already exist, see the orth-vs-O-LoRA campaign).
Every OTHER cell, including the same (0.5, 10) point on cifar224 (no existing
orth CIFAR-100 result), is generated fresh.

THREE ABLATIONS (at the baseline cell: svd_rank=10, align_weight=0.5), each
new relative to prior results:
  - countsketch_orth: merge_op=countsketch + orth. Prior CountSketch result
    (exps/sketchlora_ablations_*/sketchlora_countsketch_*) is non-orth
    (model_name=sketchlora) -- this is the same merge algorithm WITH the
    orth regularizer on top.
  - exactsvd_orth: merge_op=exactsvd + orth (full torch.linalg.svd instead
    of randomized).
  - randomsel_orth: merge_op=randsvd, sketchlora_rank_selection="random"
    (2026-09-02 NEW code, models/sketchlora.py + utils/randsvd.py) -- instead
    of keeping the r_hat_t LARGEST singular values every merge, keeps a
    random r_hat_t-sized subset of the composite's meaningful directions.
    Built to empirically test whether keeping the largest values specifically
    is what makes truncation good, vs. any fixed-size subset being
    sufficient -- see that flag's __init__ docstring in models/sketchlora.py.
The fixed-rank NON-orth SketchLoRA ablation is intentionally NOT included --
already have that result on both datasets per 2026-09-02 user confirmation.

sketch_diag=True + a per-subdir sketchlora_diag_dir (same convention as
scripts/gen_sketchlora_ablations_and_sens_configs.py) to avoid the sketch_diag
filename collision that convention exists to prevent when multiple campaigns
share a (dataset, rank, seed) cell.
"""
import json
import os

OUT_BASE = "exps/sketchlora_orth_sens_ablations"
RUN_LOGS_BASE = "run_logs/sketchlora_orth_sens_ablations"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
}

# imagenetr's (0.5, 10) cell already exists -- see module docstring.
REUSE_EXISTING = {
    ("imagenetr", 0.5, 10),
}

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
    merge_op="randsvd",
    device=["0"],
)


def write_cfg(subdir, ds_name, variant, seed, overrides):
    cfg = dict(BASE)
    cfg.update(DATASETS[ds_name])
    cfg["seed"] = [seed]
    cfg.update(overrides)
    cfg["prefix"] = "sketchlora_orth_sens_ablations_{}_{}_{}_s{}".format(
        subdir, ds_name, variant, seed)
    # forward slashes explicit (not os.path.join) -- built on Windows, read
    # back by main.py on the Linux cluster (same convention/reasoning as
    # gen_sketchlora_ablations_and_sens_configs.py).
    cfg["sketchlora_diag_dir"] = "{}/{}/{}/diag".format(RUN_LOGS_BASE, subdir, ds_name)
    outdir = os.path.join(OUT_BASE, subdir, ds_name)
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


ABLATIONS = {
    "countsketch_orth": dict(merge_op="countsketch"),
    "exactsvd_orth": dict(merge_op="exactsvd"),
    "randomsel_orth": dict(merge_op="randsvd", sketchlora_rank_selection="random"),
}


def main():
    written = []
    skipped_reuse = []

    for ds_name in DATASETS:
        for (weight, rank), variant in sensitivity_cells().items():
            if (ds_name, weight, rank) in REUSE_EXISTING:
                skipped_reuse.append((ds_name, variant))
                continue
            for seed in SEEDS:
                written.append(write_cfg("sensitivity", ds_name, variant, seed, dict(
                    svd_rank=rank, sketchlora_align_weight=weight,
                )))

        for variant, overrides in ABLATIONS.items():
            for seed in SEEDS:
                cfg_overrides = dict(svd_rank=10, sketchlora_align_weight=0.5)
                cfg_overrides.update(overrides)
                written.append(write_cfg("ablations", ds_name, variant, seed, cfg_overrides))

    print("wrote {} configs under {}/".format(len(written), OUT_BASE))
    for ds_name, variant in skipped_reuse:
        print("  reused existing result, not regenerated: {} / {}".format(ds_name, variant))


if __name__ == "__main__":
    main()
