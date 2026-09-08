"""Full (orth) SketchLoRA across all 5 vision benchmarks (2026-09-07 user
request), using a ViT-B/16 backbone pretrained on ImageNet-21k ONLY -- no
ImageNet-1k fine-tuning -- instead of this project's usual default backbone.
15 runs (5 datasets x 3 seeds). model_name=sketchlora_align,
sketchlora_align_mode=orth, sketchlora_align_weight=0.5, svd_rank=10 (the
standing "SketchLoRA" default -- unqualified "SketchLoRA" means this orth
variant, per project convention; see models/sketchlora_align.py and the
sketchlora-orth-is-now-default-convention note).

BACKBONE: backbone_type="vit_base_patch16_224_in21k_lora" (backbone/vit_lora.py,
already implemented, no new backbone code needed) resolves via timm to Google's
AugReg checkpoint B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz
-- confirmed directly against timm's registered default_cfgs (num_classes=21843,
no "--imagenet2012-..." fine-tuning suffix, unlike the project's usual
"vit_base_patch16_224" backbone which IS further fine-tuned on ImageNet-1k).
Same AugReg pretraining recipe/epoch count as the usual backbone, differing
ONLY in whether the ImageNet-1k fine-tuning stage happened.

OmniBenchmark-1k is INCLUDED here (unlike sketchlora_orth_wave1_datasets,
which excluded it as already covered) -- this is a genuinely new axis
(backbone pretraining), not a duplicate of existing OmniBenchmark-1k data,
which was all run on the usual fine-tuned backbone.

Per-dataset split params copied verbatim from exps/wave1_final/olora_<dataset>_
s1993.json (same convention as sketchlora_orth_wave1_datasets):
  cifar224:       init_cls=10, increment=10  -> 10 tasks (100 classes)
  imagenetr:      init_cls=10, increment=10  -> 20 tasks (200 classes)
  omnibenchmark1k: init_cls=10, increment=10 -> 100 tasks (1000 classes)
  food101:        init_cls=6,  increment=5   -> 20 tasks (101 classes)
  sun397:         init_cls=37, increment=40  -> 10 tasks (397 classes)
"""
import json
import os

OUT_DIR = "exps/sketchlora_orth_in21k_5datasets"
RUN_LOGS_BASE = "run_logs/sketchlora_orth_in21k_5datasets"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "omnibenchmark1k": dict(dataset="omnibenchmark1k", init_cls=10, increment=10),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
}

BASE = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora_align", backbone_type="vit_base_patch16_224_in21k_lora",
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
            cfg["prefix"] = "sketchlora_orth_in21k_5datasets_{}_s{}".format(ds_name, seed)
            cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, ds_name)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(ds_name, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
