"""Config generator for the 50MB/15-task local anchor grid (user request
2026-07-27): SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA, all at
bm_budget_mb=50, stop_after_tasks=15, OmniBenchmark-1K, single seed 1993.

Per-method HPs match the project's settled conventions (identical to
scripts/gen_round2_slurm_grid_configs.py's METHOD_CFG / the pre-existing
exps/round2_anchor/{inflora,seqlora}_50mb_15t.json, which this run reruns
fresh alongside the 3 new methods). InfLoRA uses the default lamb=0.95/
lame=1.0 ramp (matching the existing anchor precedent), NOT the H200 grid's
constant-0.975 setting -- this is a rerun/extension of the anchor family, not
the H200 grid.

"device" is left as the literal string "PLACEHOLDER", substituted per-worker
by scripts/round2_anchor_15t_queue.sh (same convention as the existing
round2_grid queue watcher scripts).
"""
import json
import os

OUT_DIR = "exps/round2_anchor"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(
    dataset="omnibenchmark1k",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, seed=[1993], scenario="cil",
    pretrained=True, print_forget=False, final_metrics=False,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    boundary_mode="bounded_memory",
    bm_budget_mb=50, stop_after_tasks=15,
    device=["PLACEHOLDER"],
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
    "inflora": dict(model_name="inflora", lora_merge=True, lamb=0.95, lame=1.0, init_lr=0.0005),
    "treelora": dict(model_name="treelora", reg=0.1, init_lr=0.001),
}

for method, cfg_extra in METHOD_CFG.items():
    cfg = dict(BASE)
    cfg.update(cfg_extra)
    cfg["prefix"] = "round2_anchor_omni15t_50mb_{}".format(method)
    path = os.path.join(OUT_DIR, "{}_50mb_15t.json".format(method))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)
