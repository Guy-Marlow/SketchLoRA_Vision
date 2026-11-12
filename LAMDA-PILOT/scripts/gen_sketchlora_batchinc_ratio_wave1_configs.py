"""SketchLoRA batch-incremental + KL-drift-detection sketch-interval ratio
sweep (2026-11-12 user request): CIFAR-100-10t and ImageNet-R-20t, full
task count (no stop_after_tasks -- this is a real analysis campaign, not a
truncated smoke test), 5 sketch-interval ratios (0.5/0.75/1.0/1.25/1.5 of
task 0's own step count) x 3 seeds (1993/1996/1999) = 2 x 5 x 3 = 30 runs.

boundary_mode=="kl_batch_inc" throughout -- the REAL KL drift detector
governs adapter spawning in every run (batchinc_fixed_interval_only is
left at its default False), independent of the sketch-interval clock
being swept here. This matches the local 6-task ImageNet-R smoke
experiment this session already ran at ratios 0.75/1.0/1.5 (this is that
same design, generalized to the full task count, 2 datasets, and the two
additional ratios 0.5/1.25).

HYPERPARAMETERS: identical to this project's own settled SketchLoRA
convention (exps/sketchlora_qrsvd_fillout/main5/<dataset>/<dataset>_
s1993.json) for everything the two settings share -- init_lr=0.001 (SAME
value for both cifar224 and imagenetr; unlike SD-LoRA/LoRA-Subtraction,
SketchLoRA's settled LR does not vary per dataset), weight_decay=0.0005,
batch_size=48, tuned_epoch=20, lora_rank=10, svd_rank=10,
batchinc_orth_weight=0.5 (matches sketchlora_align_weight=0.5). The
detector/fold-clock hyperparameters (batchinc_pending_cap,
batchinc_kl_window_samples/_tau/_consecutive/_warmup_multiplier,
batchinc_lr_warmup_steps) have no "settled SketchLoRA" equivalent -- they
are specific to this setting and reuse the exact values already
validated this session's local smoke runs (CIFAR 3-task post-fix
validation + ImageNet-R 6-task ratio runs), not re-tuned here.

SKETCH-INTERVAL-STEPS, per dataset: computed from task 0's own measured
step count at this batch_size/epoch count (NOT a fixed-images-per-class
assumption -- measured directly via utils.data_manager.DataManager, since
per-task image counts are not equal across tasks or datasets):
  - cifar224 task 0: 5000 images, 105 batches/epoch x 20 epochs = 2100 steps.
  - imagenetr task 0: 1289 images, 27 batches/epoch x 20 epochs = 540 steps.
batchinc_sketch_interval_steps = round(ratio * task0_steps) for each of
the 5 ratios.

OUTPUT: boundary_mode=="kl_batch_inc" does not use the run_logs/final/
<model>/metrics_*.json / final_metrics convention at all -- it writes
run_logs/kl_batch_inc_<model_name>_<prefix>_s<seed>.json via trainer.py::
_run_stream (see scripts/_kl_batch_inc_run_done.py, the resumability
check this campaign's SLURM script uses INSTEAD of scripts/
_bankcap_run_done.py). `prefix` is unique per (dataset, ratio, seed) --
see PREFIX below -- so every run's output path is distinct and nothing
here can collide with or overwrite the earlier local 6-task smoke runs'
own output files (different prefixes) or each other.
"""
import json
import os

OUT_BASE = "exps/sketchlora_batchinc_ratio_wave1"
SEEDS = [1993, 1996, 1999]
RATIOS = [0.5, 0.75, 1.0, 1.25, 1.5]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10, task0_steps=2100),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10, task0_steps=540),
}

COMMON = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=False,
    boundary_mode="kl_batch_inc", model_name="sketchlora_batchinc",
    backbone_type="vit_base_patch16_224_sketchlora_batchinc",
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    lora_rank=10, lora_alpha=None, svd_rank=10,
    batchinc_pending_cap=3, batchinc_fixed_interval_only=False,
    batchinc_orth_weight=0.5, batchinc_kl_window_samples=256, batchinc_kl_tau=4.0,
    batchinc_kl_consecutive=3, batchinc_kl_warmup_multiplier=2, batchinc_lr_warmup_steps=20,
    device=["0"],
)


def cfg_for(ds_name, ratio, seed):
    ds = DATASETS[ds_name]
    interval = round(ratio * ds["task0_steps"])
    cfg = dict(COMMON)
    cfg.update(dataset=ds["dataset"], init_cls=ds["init_cls"], increment=ds["increment"])
    cfg.update(
        seed=[seed], batchinc_sketch_interval_steps=interval,
        prefix="sketchlora_batchinc_ratio_wave1_{}_ratio{}_s{}".format(ds_name, ratio, seed),
    )
    return cfg


def main():
    n = 0
    for ds_name in DATASETS:
        for ratio in RATIOS:
            out_dir = os.path.join(OUT_BASE, ds_name, "ratio{}".format(ratio))
            os.makedirs(out_dir, exist_ok=True)
            for seed in SEEDS:
                cfg = cfg_for(ds_name, ratio, seed)
                out_path = os.path.join(out_dir, "s{}.json".format(seed))
                json.dump(cfg, open(out_path, "w"), indent=2)
                n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
