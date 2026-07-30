"""Config generator for the ImageNet-R local grid (user request 2026-07-28):
SeqLoRA, O-LoRA, InfLoRA, TreeLoRA, SketchLoRA (added later the same day, runs
AFTER TreeLoRA in method order per the user's spec) x {50, 100, 200}MB x 2
seeds (1993, 1996), full 20-task ImageNet-R split, bounded_memory boundary
mode, CE metric always on (models/bounded_memory_mixin.py now wires it in
unconditionally; SketchLoRA's own CE aux-cost hooks -- sketch-inclusion
forward + per-fold merge + CA -- were added to models/sketchlora.py at the
same time so its CE reading isn't misleadingly 1.0 like a zero-overhead
method). SketchLoRA uses the project's default "no-shrink" admission rule
(sketchlora_admission="bounded_eviction") with sketchlora_rank_cap=128, same
convention as every other SketchLoRA run this session.

HP conventions: rank-10/alpha-null (scaling=1) for EVERY method, matching the
H200 cluster grid and the InfLoRA lamb/lame sensitivity run exactly (user-
confirmed 2026-07-28: "we are doing 1x scaling, with rank 10 adapters, as we
should be doing with every single other test... that's the setting that's
running on the cluster" -- verified directly against exps/sensitivity_inflora_
lamblame0975_100mb_s1993.json, which is rank10/alpha-null). An EARLIER draft
of this script used rank8/alpha32 per a since-superseded memory note (2026-
07-01 CIFAR-era convention) -- corrected before any real run launched; see
the vision_rank8_scaling4_convention memory file for the full correction.
Per-method optimizer/regularizer HPs otherwise match this project's settled
conventions (scripts/gen_round2_slurm_grid_configs.py), with InfLoRA using a
CONSTANT lamb=lame=0.975 (matches the H200 grid and the sensitivity run).
"""
import json
import os

OUT_DIR = "exps/imagenetr_grid"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(
    dataset="imagenetr",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, scenario="cil",
    pretrained=True, print_forget=False, final_metrics=False,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    boundary_mode="bounded_memory",
    device=["PLACEHOLDER"],
)

METHOD_CFG = {
    "seqlora": dict(model_name="seqlora", lora_merge=False, init_lr=0.0003),
    "olora": dict(model_name="olora", lora_merge=True, lamda_1=0.5, lamda_2=0.0, init_lr=0.001),
    "inflora": dict(model_name="inflora", lora_merge=True, lamb=0.975, lame=0.975, init_lr=0.0005),
    "treelora": dict(model_name="treelora", reg=0.1, init_lr=0.001),
    "sketchlora": dict(
        model_name="sketchlora", lora_merge=True, lora_train_merge=True,
        svd_rank=10, svd_oversampling=10, svd_energy_target=0.01,
        lora_n_slots=2, sketch_diag=True, merge_op="randsvd",
        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        sketchlora_lora_wd=0.0, sketchlora_eviction_reading="conformant",
        init_lr=0.001,
    ),
}

# order matters -- method cycling order within a (budget, seed) group
METHODS = ["seqlora", "olora", "inflora", "treelora", "sketchlora"]
BUDGETS = [50, 100, 200]     # order matters -- outer loop
SEEDS = [1993, 1996]         # order matters -- inner loop within a budget

configs = []
for budget in BUDGETS:
    for seed in SEEDS:
        for method in METHODS:
            cfg = dict(BASE)
            cfg.update(METHOD_CFG[method])
            cfg["seed"] = [seed]
            cfg["bm_budget_mb"] = budget
            cfg["prefix"] = "imagenetr_grid_{}mb_{}_s{}".format(budget, method, seed)
            fname = "{}_{}mb_s{}.json".format(method, budget, seed)
            path = os.path.join(OUT_DIR, fname)
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    print(c)
