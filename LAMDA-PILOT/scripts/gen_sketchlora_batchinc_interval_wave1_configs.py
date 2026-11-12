"""SketchLoRA batch-incremental + KL-drift-detection sketch-interval sweep
(2026-11-12 user request, revised from an earlier ratio-of-task0 design to
ABSOLUTE step-count increments, per-dataset): CIFAR-100-10t and
ImageNet-R-20t, full task count (no stop_after_tasks -- a real analysis
campaign, not a truncated smoke test), 5 fixed sketch-interval-step values
per dataset x 3 seeds (1993/1996/1999) = 2 x 5 x 3 = 30 runs.

STEP INCREMENTS (absolute batchinc_sketch_interval_steps, not a ratio of
task 0's own step count -- that was this campaign's original design,
replaced on explicit instruction):
  - imagenetr: 200, 400, 600, 800, 1000 -- comparable to task 0's own 540
    steps and the dataset's mean ~509.7 steps/task across all 20 tasks x 3
    seeds (measured directly via utils.data_manager.DataManager), so this
    range spans well below to well above a typical task's length.
  - cifar224: 1000, 1500, 2000, 2500, 3000 -- task 0 (and every task, since
    CIFAR-100 is exactly 500 images/class x 10 classes/task = 5000 images
    every task) is 2100 steps at this batch_size/epoch count, so this
    range spans below to above that fixed per-task length too.

boundary_mode=="kl_batch_inc" throughout -- the REAL KL drift detector
governs adapter spawning in every run (batchinc_fixed_interval_only is
left at its default False), independent of the sketch-interval clock
being swept here.

HYPERPARAMETERS: identical to this project's own settled SketchLoRA
convention (exps/sketchlora_qrsvd_fillout/main5/<dataset>/<dataset>_
s1993.json) for everything the two settings share -- init_lr=0.001 (SAME
value for both cifar224 and imagenetr), weight_decay=0.0005,
batch_size=48, tuned_epoch=20, lora_rank=10, svd_rank=10,
batchinc_orth_weight=0.5 (matches sketchlora_align_weight=0.5). The
detector/fold-clock hyperparameters (batchinc_pending_cap,
batchinc_kl_window_samples/_tau/_consecutive/_warmup_multiplier,
batchinc_lr_warmup_steps) have no "settled SketchLoRA" equivalent -- they
reuse the exact values already validated this session's local smoke runs,
not re-tuned here.

OUTPUT: boundary_mode=="kl_batch_inc" does not use the run_logs/final/
<model>/metrics_*.json / final_metrics convention at all -- it writes
run_logs/kl_batch_inc_<model_name>_<prefix>_s<seed>.json via trainer.py::
_run_stream (see scripts/_kl_batch_inc_run_done.py, the resumability check
this campaign's SLURM script uses INSTEAD of scripts/_bankcap_run_done.py).
`prefix` is unique per (dataset, interval, seed), so every run's output
path is distinct.
"""
import json
import os

OUT_BASE = "exps/sketchlora_batchinc_interval_wave1"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10,
                      intervals=[1000, 1500, 2000, 2500, 3000]),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10,
                       intervals=[200, 400, 600, 800, 1000]),
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


def cfg_for(ds_name, interval, seed):
    ds = DATASETS[ds_name]
    cfg = dict(COMMON)
    cfg.update(dataset=ds["dataset"], init_cls=ds["init_cls"], increment=ds["increment"])
    cfg.update(
        seed=[seed], batchinc_sketch_interval_steps=interval,
        prefix="sketchlora_batchinc_interval_wave1_{}_interval{}_s{}".format(ds_name, interval, seed),
    )
    return cfg


def main():
    n = 0
    for ds_name, ds in DATASETS.items():
        for interval in ds["intervals"]:
            out_dir = os.path.join(OUT_BASE, ds_name, "interval{}".format(interval))
            os.makedirs(out_dir, exist_ok=True)
            for seed in SEEDS:
                cfg = cfg_for(ds_name, interval, seed)
                out_path = os.path.join(out_dir, "s{}.json".format(seed))
                json.dump(cfg, open(out_path, "w"), indent=2)
                n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
