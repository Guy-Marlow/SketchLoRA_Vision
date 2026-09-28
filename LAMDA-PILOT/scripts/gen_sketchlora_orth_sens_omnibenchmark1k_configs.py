"""SketchLoRA orth sensitivity campaign, extended to OmniBenchmark-1k's
100-task split (2026-09-28 user request) -- the same two-axis sensitivity
grid scripts/gen_sketchlora_orth_sens_ablations_configs.py already ran on
ImageNet-R-20t/CIFAR-100-10t, now on OmniBenchmark-1k-100t:
  - orth strength: sketchlora_align_weight in {0.1, 0.3, 0.7, 0.9},
    svd_rank fixed at 10.
  - sketch rank: svd_rank in {5, 15, 20, 25}, align_weight fixed at 0.5.
The baseline cell (svd_rank=10, align_weight=0.5) is NOT regenerated here --
already have it byte-for-byte as
exps/sketchlora_orth_omnibenchmark1k/sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s<seed>.json
(the source this campaign's hyperparameters are copied from). 8 cells x 3
seeds = 24 runs.

Every other hyperparameter matches that baseline exactly (tuned_epoch=20,
batch_size=48, init_lr=0.001, lora_rank=10, svd_oversampling=10,
lora_n_slots=2, merge_op=randsvd, align_mode=orth) -- only svd_rank/
sketchlora_align_weight/seed/prefix/sketchlora_diag_dir vary.
"""
import json
import os

OUT_DIR = "exps/sketchlora_orth_sens_omnibenchmark1k"
RUN_LOGS_BASE = "run_logs/sketchlora_orth_sens_omnibenchmark1k"
SEEDS = [1993, 1996, 1999]

BASE = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora_align", backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    dataset="omnibenchmark1k", init_cls=10, increment=10,
    lora_merge=True, lora_train_merge=True,
    svd_oversampling=10, lora_n_slots=2, sketch_diag=True,
    sketchlora_lora_wd=0.0,
    sketchlora_align_mode="orth",
    merge_op="randsvd",
    device=["0"],
)


def sensitivity_cells():
    """8-point (weight, rank) grid -- the baseline (0.5, 10) cell is excluded
    (already have that result), unlike gen_sketchlora_orth_sens_ablations_
    configs.py's 9-point grid which keeps it for the ImageNet-R/CIFAR case."""
    cells = {}
    for w in (0.1, 0.3, 0.7, 0.9):
        cells[(w, 10)] = "w{}_r10".format(w)
    for r in (5, 15, 20, 25):
        cells[(0.5, r)] = "w0.5_r{}".format(r)
    return cells


def write_cfg(variant, seed, overrides):
    cfg = dict(BASE)
    cfg["seed"] = [seed]
    cfg.update(overrides)
    cfg["prefix"] = "sketchlora_orth_sens_omnibenchmark1k_{}_s{}".format(variant, seed)
    # forward slashes explicit (not os.path.join) -- built on Windows, read
    # back by main.py on the Linux cluster (same convention as
    # gen_sketchlora_orth_sens_ablations_configs.py).
    cfg["sketchlora_diag_dir"] = "{}/diag".format(RUN_LOGS_BASE)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "{}_s{}.json".format(variant, seed))
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def main():
    written = []
    for (weight, rank), variant in sensitivity_cells().items():
        for seed in SEEDS:
            written.append(write_cfg(variant, seed, dict(
                svd_rank=rank, sketchlora_align_weight=weight,
            )))
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
