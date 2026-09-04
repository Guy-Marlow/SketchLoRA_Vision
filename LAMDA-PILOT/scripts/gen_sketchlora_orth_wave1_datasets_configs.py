"""Full (orth) SketchLoRA across the wave1 dataset splits, minus OmniBenchmark-1k
(2026-09-04 user request -- "complete our large master table with updated
SketchLoRA results"). 4 datasets x 3 seeds = 12 configs. model_name=
sketchlora_align, sketchlora_align_mode=orth, sketchlora_align_weight=0.5,
svd_rank=10 (the standing "SketchLoRA" default -- see models/sketchlora_align.py
and the sketchlora-orth-is-now-default-convention note).

Per-dataset split params copied VERBATIM from exps/wave1_final/olora_<dataset>_
s1993.json (same init_cls/increment every other wave1_final method uses for
that dataset -- this campaign must be comparable to the rest of the master
table, not an independently-invented split):
  cifar224:  init_cls=10, increment=10  -> 10 tasks (100 classes)
  imagenetr: init_cls=10, increment=10  -> 20 tasks (200 classes)
  food101:   init_cls=6,  increment=5   -> 20 tasks (101 classes) -- verified
             6 + 5*19 = 101 exactly, no remainder.
  sun397:    init_cls=37, increment=40  -> 10 tasks (397 classes) -- VERIFIED
             2026-09-04 per explicit user request to double-check this: 37 +
             40*9 = 397 exactly, no remainder. Already a clean 10-task split
             under wave1_final's own existing config; nothing needed fixing.
OmniBenchmark-1k intentionally excluded -- already have that result.

NOTE: this reruns ImageNet-R-20t fresh under this campaign's own prefix, even
though a 3-seed (svd_rank=10, align_weight=0.5) orth result already exists
locally (exps/sketchlora_fixedrank_orth05_imagenetr_s*.json, reused instead
of rerun for the sensitivity/ablation campaign). Included here anyway so all
4 master-table cells come from one uniform, consistently-configured
campaign -- flag if you'd rather this be excluded and the existing result
reused instead.
"""
import json
import os

OUT_DIR = "exps/sketchlora_orth_wave1_datasets"
RUN_LOGS_BASE = "run_logs/sketchlora_orth_wave1_datasets"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
}

BASE = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora_align", backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    lora_merge=True, lora_train_merge=True,
    svd_rank=10, svd_oversampling=10, lora_n_slots=2, sketch_diag=True,
    sketchlora_lora_wd=0.0,
    sketchlora_align_mode="orth", sketchlora_align_weight=0.5,
    merge_op="randsvd",
    device=["0"],
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for ds_name, ds_overrides in DATASETS.items():
        for seed in SEEDS:
            cfg = dict(BASE)
            cfg.update(ds_overrides)
            cfg["seed"] = [seed]
            cfg["prefix"] = "sketchlora_orth_wave1_datasets_{}_s{}".format(ds_name, seed)
            cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, ds_name)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(ds_name, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
