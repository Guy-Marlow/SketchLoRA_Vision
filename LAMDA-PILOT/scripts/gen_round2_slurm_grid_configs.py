"""Config generator for the round-2 SLURM grid (user request 2026-07-27):
SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA x {100, 200}MB x 3 seeds,
OmniBenchmark-1K, 30 latent tasks, bounded_memory boundary mode -- same
harness/fixes as the round-2 dose-response grid (exps/round2_grid/), just a
different method/budget/seed subset plus TreeLoRA (new to round 2) and a
frozen (non-ramped) InfLoRA retention setting.

Per-method HP conventions (identical to exps/round2_grid/ and
scripts/run_lr_sweep_b2.py's CIFAR-100 sweep winners, propagated per the
project's own CIFAR-winner -> Omni rule):
  seqlora:    lr=3e-4 (no merge)
  olora:      lr=1e-3, lamda_1=0.5, lamda_2=0.0
  inflora:    lr=5e-4, lamb=lame=0.975 (CONSTANT threshold, user-specified --
              see docs/plan_c_bounded_memory_round2.md's sensitivity-run
              writeup for why: removes InfLoRA's total-stream-length
              dependence entirely, since lamb=lame cancels the cur_task/
              total_sessions ramp term out of the threshold formula)
  treelora:   lr=1e-3, reg=0.1 (matches run_lr_sweep_b2.py's METHOD_CFG;
              the code's own default is reg=0.5, but 0.1 is the value this
              project actually swept/validated under, so that's what's used
              here -- flagging the discrepancy rather than silently picking
              one)
  sketchlora: lr=1e-3 (round-2 3.1 sweep winner), svd_energy_target=0.01,
              merge_op=randsvd with svd_oversampling=10 (REVERTED 2026-07-27,
              user-specified: RandSVD is central to the method's guarantees
              and its incurred error is low -- supersedes round-2 2.5's
              temporary "exact SVD until the A5.3 decision is cleared"
              fallback; this SLURM grid is the actual production RandSVD
              setting, not a debugging substitute), rank_cap=128,
              bounded_eviction/conformant (round-2 2.4), lora_wd=0

Device: every config uses "device": ["0"] uniformly -- physical GPU
selection happens entirely via CUDA_VISIBLE_DEVICES in the .slurm script's
worker lanes (mirrors scripts/final_vision.slurm's convention), not via the
config's own device field. This sidesteps the CUDA_DEVICE_ORDER/PCI-bus
device-index gotcha entirely (see docs/plan_c_bounded_memory_round2.md) since
CUDA_VISIBLE_DEVICES restricts visibility before any index ambiguity can
occur.
"""
import json
import os

# Relative to cwd deliberately -- this script is invoked by round2_slurm_grid.slurm
# AFTER it cd's into the project root, and that root's absolute path differs
# between the testing cluster (4 A100s, interactive, /home/gmar762/...) and the
# experiments cluster (8 H200s, SLURM, /data/gmar762/... -- see
# scripts/round2_slurm_grid.slurm's own VISION variable). No absolute path here
# avoids hardcoding either.
OUT_DIR = "exps/round2_slurm_grid"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(
    dataset="omnibenchmark1k",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, scenario="cil",
    pretrained=True, print_forget=False, final_metrics=False,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    boundary_mode="bounded_memory",
    stop_after_tasks=30,
    device=["0"],
)

METHOD_CFG = {
    "seqlora": dict(model_name="seqlora", lora_merge=False, init_lr=0.0003),
    "sketchlora": dict(
        model_name="sketchlora", lora_merge=True, lora_train_merge=True,
        svd_rank=10, svd_oversampling=10, svd_energy_target=0.01,
        lora_n_slots=2, sketch_diag=True, merge_op="randsvd",
        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        sketchlora_lora_wd=0.0, sketchlora_eviction_reading="conformant",
        init_lr=0.001,
    ),
    "olora": dict(model_name="olora", lora_merge=True, lamda_1=0.5, lamda_2=0.0, init_lr=0.001),
    "inflora": dict(model_name="inflora", lora_merge=True, lamb=0.975, lame=0.975, init_lr=0.0005),
    "treelora": dict(model_name="treelora", reg=0.1, init_lr=0.001),
}

# order matters -- METHOD-MAJOR execution order in the .slurm script
METHODS = ["seqlora", "sketchlora", "olora", "inflora", "treelora"]
BUDGETS = [100, 200]   # order matters -- outer loop in the .slurm script
SEEDS = [1993, 1996, 1999]

configs = []
for budget in BUDGETS:
    for method in METHODS:
        for seed in SEEDS:
            cfg = dict(BASE)
            cfg.update(METHOD_CFG[method])
            cfg["seed"] = [seed]
            cfg["bm_budget_mb"] = budget
            cfg["prefix"] = "round2_slurm_grid_omni30t_{}mb_{}_s{}".format(budget, method, seed)
            fname = "{}_{}mb_s{}.json".format(method, budget, seed)
            path = os.path.join(OUT_DIR, fname)
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    print(c)
