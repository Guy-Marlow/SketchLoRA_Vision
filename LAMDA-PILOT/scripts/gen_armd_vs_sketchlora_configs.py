"""Generates the armd_vs_sketchlora campaign: three method arms x 3 seeds x
3 datasets = 27 runs.

  1. A-only fixed-rank orthogonalized SketchLoRA (sketchlora_align,
     align_mode=orth, weight=0.5, svd_rank pinned at 10, no adaptive/retain
     admission).
  2. Arm D O-LoRA (real O-LoRA, bank_cap_mb=3.0 -- admits exactly 2 slots
     regardless of dataset since LoRA slot size is a pure architecture
     constant (rank=10, dim=768, 24 wrapped modules), never dataset-
     dependent -- merge left at its default True, i.e. NOT the nomerge
     ablation override, so slot 0 stays folded into every forward exactly
     like normal O-LoRA; see models/olora.py's own bank-cap pinning logic
     for why this collapses to "orthogonalize only against task 0, but keep
     the merge" once the 3rd-slot admission is refused).
  3. Adaptive-rank orthogonalized SketchLoRA (sketchlora_align,
     sketchlora_admission="retain", svd_energy_target=0.25 -- i.e. keep 75%
     of the weight/energy of each newly-introduced adapter's own
     sub-spectrum, per task, same "retain" mechanism as the earlier
     sketchlora_retain_fixed09 run: unconditionally keeps the sketch's
     entire existing rank, only the new residual's own energy is
     thresholded -- align_mode=orth, weight=0.5 (2026-08-23 user request:
     higher weight than the earlier 0.25 retain run), rank_cap=128 as the
     hard ceiling.

3 seeds (1993/1996/1999) x 3 datasets:
  - cifar224: init_cls=10/increment=10, naturally 10 tasks (no truncation
    needed) -- matches wave1_final's own cifar224 shape exactly.
  - imagenetr: init_cls=10/increment=10 (wave1_final's native 20-task split),
    stop_after_tasks=10 -- "imagenetr10t", matching the naming/truncation
    convention already used elsewhere in this project (e.g.
    exps/treelora_routing_fix_check/treelora_imagenetr10t_*.json).
  - food101: init_cls=6/increment=5, naturally 20 tasks -- matches
    wave1_final's own food101 shape exactly (no "-10t" requested for this
    one, so its native shape is used unchanged).

Hyperparams (tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=
0.0005, lora_rank=10) match every wave1_final config across all three
datasets -- confirmed identical dataset-to-dataset in the existing configs
before reusing them here, not assumed.
"""
import json
import os

OUT_DIR = "exps/armd_vs_sketchlora"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224_10t": dict(dataset="cifar224", init_cls=10, increment=10, stop_after_tasks=None),
    "imagenetr10t": dict(dataset="imagenetr", init_cls=10, increment=10, stop_after_tasks=10),
    "food101": dict(dataset="food101", init_cls=6, increment=5, stop_after_tasks=None),
}

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


def sketchlora_fixedrank_orth_config(tag, ds, seed):
    cfg = dict(COMMON)
    cfg.update(ds)
    cfg["seed"] = [seed]
    cfg["model_name"] = "sketchlora_align"
    prefix = "armd_vs_sketchlora_fixedrank_orth05_{}_s{}".format(tag, seed)
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
        sketchlora_diag_dir="run_logs/armd_vs_sketchlora/{}/sketch_diag".format(tag),
    )
    if ds["stop_after_tasks"] is None:
        del cfg["stop_after_tasks"]
    return prefix, cfg


def sketchlora_adaptive_orth_config(tag, ds, seed):
    cfg = dict(COMMON)
    cfg.update(ds)
    cfg["seed"] = [seed]
    cfg["model_name"] = "sketchlora_align"
    prefix = "armd_vs_sketchlora_adaptive_orth05_eps025_{}_s{}".format(tag, seed)
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
        sketchlora_diag_dir="run_logs/armd_vs_sketchlora/{}/sketch_diag_adaptive".format(tag),
    )
    if ds["stop_after_tasks"] is None:
        del cfg["stop_after_tasks"]
    return prefix, cfg


def armd_olora_config(tag, ds, seed):
    cfg = dict(COMMON)
    cfg.update(ds)
    cfg["seed"] = [seed]
    cfg["model_name"] = "olora"
    prefix = "armd_vs_sketchlora_armD_olora_cap3_{}_s{}".format(tag, seed)
    cfg["prefix"] = prefix
    cfg.update(
        lora_merge=True,
        lamda_1=0.5,
        lamda_2=0.0,
        bank_cap_mb=3.0,
        # lora_train_merge intentionally NOT set -- defaults to True (real
        # O-LoRA's own merge behavior), unlike the olora_mechanism_ablation
        # campaign's Arm B, which explicitly set this to False.
    )
    if ds["stop_after_tasks"] is None:
        del cfg["stop_after_tasks"]
    return prefix, cfg


def main():
    written = []
    for tag, ds in DATASETS.items():
        for seed in SEEDS:
            for builder in (sketchlora_fixedrank_orth_config, armd_olora_config,
                            sketchlora_adaptive_orth_config):
                prefix, cfg = builder(tag, ds, seed)
                path = os.path.join(OUT_DIR, prefix + ".json")
                with open(path, "w") as f:
                    json.dump(cfg, f, indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
