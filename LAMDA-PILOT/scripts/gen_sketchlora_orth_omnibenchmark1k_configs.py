"""Generates the sketchlora_orth_omnibenchmark1k campaign: the same two
orthogonalized-SketchLoRA arms from armd_vs_sketchlora (cifar224/imagenetr10t/
food101), extended to OmniBenchmark-1k, 3 seeds each = 6 runs.

  1. Fixed-rank orthogonalized SketchLoRA (sketchlora_align, align_mode=orth,
     weight=0.5, svd_rank pinned at 10, no adaptive/retain admission).
  2. Adaptive-rank orthogonalized SketchLoRA (sketchlora_align,
     sketchlora_admission="retain", svd_energy_target=0.25 -- keep 75% of
     each newly-introduced adapter's own energy per task, sketch's existing
     rank never evicted -- align_mode=orth, weight=0.5, rank_cap=128).

Both otherwise byte-identical to their armd_vs_sketchlora counterparts --
same hyperparameters, same mechanism -- just pointed at omnibenchmark1k
(init_cls=10/increment=10 on the 1000-class set -> 100 tasks, no truncation,
matching wave1_final's own omnibenchmark1k shape and SketchLoRA's established
lr=0.001 there) instead of the three smaller datasets. bank_cap_mb
intentionally unset on both (uncapped), matching armd_vs_sketchlora's
SketchLoRA arms exactly -- capping was only ever the O-LoRA arm's thing.
"""
import json
import os

OUT_DIR = "exps/sketchlora_orth_omnibenchmark1k"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1993, 1996, 1999]

DATASET = dict(dataset="omnibenchmark1k", init_cls=10, increment=10)

COMMON = dict(
    memory_size=0,
    memory_per_class=0,
    fixed_memory=False,
    shuffle=True,
    scenario="cil",
    pretrained=True,
    print_forget=True,
    final_metrics=True,
    tuned_epoch=20,
    batch_size=48,
    init_lr=0.001,
    weight_decay=0.0005,
    min_lr=0.0,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10,
    lora_alpha=None,
    device=["0"],
)


def sketchlora_fixedrank_orth_config(seed):
    cfg = dict(COMMON)
    cfg.update(DATASET)
    cfg["seed"] = [seed]
    cfg["model_name"] = "sketchlora_align"
    prefix = "sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s{}".format(seed)
    cfg["prefix"] = prefix
    cfg.update(
        lora_merge=True,
        lora_train_merge=True,
        svd_rank=10,
        svd_oversampling=10,
        lora_n_slots=2,
        sketch_diag=True,
        merge_op="randsvd",
        sketchlora_lora_wd=0.0,
        sketchlora_align_mode="orth",
        sketchlora_align_weight=0.5,
        sketchlora_diag_dir="run_logs/sketchlora_orth_omnibenchmark1k/sketch_diag_fixedrank",
    )
    return prefix, cfg


def sketchlora_adaptive_orth_config(seed):
    cfg = dict(COMMON)
    cfg.update(DATASET)
    cfg["seed"] = [seed]
    cfg["model_name"] = "sketchlora_align"
    prefix = "sketchlora_orth_omnibenchmark1k_adaptive_orth025_s{}".format(seed)
    cfg["prefix"] = prefix
    cfg.update(
        lora_merge=True,
        lora_train_merge=True,
        svd_rank=10,
        svd_oversampling=10,
        lora_n_slots=2,
        sketch_diag=True,
        merge_op="randsvd",
        svd_energy_target=0.25,
        sketchlora_admission="retain",
        sketchlora_rank_cap=128,
        sketchlora_lora_wd=0.0,
        sketchlora_align_mode="orth",
        sketchlora_align_weight=0.5,
        sketchlora_diag_dir="run_logs/sketchlora_orth_omnibenchmark1k/sketch_diag_adaptive",
    )
    return prefix, cfg


def main():
    written = []
    for seed in SEEDS:
        for builder in (sketchlora_fixedrank_orth_config, sketchlora_adaptive_orth_config):
            prefix, cfg = builder(seed)
            path = os.path.join(OUT_DIR, prefix + ".json")
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
