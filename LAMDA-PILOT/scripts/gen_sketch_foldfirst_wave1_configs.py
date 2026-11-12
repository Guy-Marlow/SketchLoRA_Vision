"""Fold-before-eval variant campaign (2026-11-11 user request): SD-LoRA-
Sketch, LoRA-Subtraction-Sketch, and CL-LoRA-Sketch, each re-run under a new
codepath (models/sdlora_sketch_foldfirst.py / lorasub_sketch_foldfirst.py /
cllora_sketch_foldfirst.py) that folds the just-finished task into its
sketch BEFORE eval_task() runs instead of after -- making eval cadence
consistent with default SketchLoRA's own convention (models/sketchlora.py's
_train() calls _compress() before returning, so eval never sees an
un-truncated residual). See each new file's own module docstring for why
this is a real accuracy-relevant change (eval currently gets a "free" boost
from the un-truncated, full-rank residual/new-adapter/block-weight state
that persistent_state() itself excludes from the counted bank) and why it
is safe to move with no other eval-routing changes (every residual/new-
adapter reset after a fold is an exact, zero-valued LoRA-style init, so the
network's own forward sum degenerates to "sketch alone" automatically).

Hyperparameters are copied byte-for-byte from this project's already-
settled conventions for each base (non-foldfirst) method -- only model_name
changes; backbone_type, architecture constants, and per-dataset LRs are
identical, since nothing about the backbone/forward mechanics changed:
  - sdlora_sketch -> sdlora_sketch_foldfirst:
    scripts/gen_lorasub_sdlora_sketch_wave1_configs.py::sdlora_sketch_cfg.
  - lorasub_sketch -> lorasub_sketch_foldfirst:
    scripts/gen_lorasub_sdlora_sketch_wave1_configs.py::lorasub_sketch_cfg.
  - cllora_sketch -> cllora_sketch_foldfirst:
    scripts/gen_bankrank_phase2_wave1_configs.py::cllora_sketch_cfg (the
    "v3" design: orth_a_side=True, use_block_weight_scaling=False,
    sketchlora_align_weight=0.75 -- the one promoted to a full campaign).

Per-dataset splits (init_cls/increment), identical to every other wave1-
style campaign in this project: cifar224=10/10 (10 tasks), imagenetr=10/10
(20 tasks), omnibenchmark1k=10/10 (100 tasks), sun397=37/40 (10 tasks),
food101=6/5 (20 tasks). 3 methods x 5 datasets x 3 seeds (1993/1996/1999) =
45 runs.
"""
import json
import os

OUT_BASE = "exps/sketch_foldfirst_wave1"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "omnibenchmark1k": dict(dataset="omnibenchmark1k", init_cls=10, increment=10),
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
}

COMMON = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, min_lr=0.0, lora_rank=10, lora_alpha=None,
    device=["0"],
)

LORASUB_SKETCH_LR = {
    "cifar224": 0.001, "imagenetr": 0.003, "omnibenchmark1k": 0.003,
    "sun397": 0.003, "food101": 0.003,
}

SDLORA_SKETCH_LR = {
    "cifar224": 0.005, "imagenetr": 0.008, "omnibenchmark1k": 0.005,
    "sun397": 0.005, "food101": 0.005,
}

CLLORA_COMMON = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    batch_size=48, weight_decay=0.0001, min_lr=0,
    init_epochs=20, later_epochs=20, init_lr=0.05, later_lr=0.05,
    optimizer="sgd", scheduler="cosine", ffn_num=10, use_distillation=True,
    alpha=0.0, beta=0,
    msa=[1, 0, 1], general_pos=[0, 1, 2, 3, 4, 5], specfic_pos=[6, 7, 8, 9, 10, 11],
    device=["0"],
)


def lorasub_sketch_foldfirst_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0,
        model_name="lorasub_sketch_foldfirst", backbone_type="vit_base_patch16_224_lorasub",
        lora_n_slots=2, svd_rank=10, merge_op="qrsvd",
        svd_thres=0.99, use_proto_repulsion=True,
        proto_repulsion_weight=0.2, proto_margin=2.0,
        init_lr=LORASUB_SKETCH_LR[ds_name],
        prefix="sketch_foldfirst_wave1_lorasub_sketch_foldfirst_{}_s{}".format(ds_name, seed),
    )
    return cfg


def sdlora_sketch_foldfirst_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0005,
        model_name="sdlora_sketch_foldfirst", backbone_type="vit_base_patch16_224_sdlora_sketch",
        svd_rank=10, sdsketch_reg_strength=0.1, sdsketch_reg_power=1.0,
        sdsketch_freeze_rescale=False,
        init_lr=SDLORA_SKETCH_LR[ds_name],
        prefix="sketch_foldfirst_wave1_sdlora_sketch_foldfirst_{}_s{}".format(ds_name, seed),
    )
    return cfg


def cllora_sketch_foldfirst_cfg(ds_name, seed):
    cfg = dict(CLLORA_COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], model_name="cllora_sketch_foldfirst",
        backbone_type="vit_base_patch16_224_cllora_sketch",
        svd_rank=10, merge_op="qrsvd",
        use_block_weight_scaling=False, orth_a_side=True,
        sketchlora_align_weight=0.75,
        prefix="sketch_foldfirst_wave1_cllora_sketch_foldfirst_{}_s{}".format(ds_name, seed),
    )
    return cfg


BUILDERS = {
    "lorasub_sketch_foldfirst": lorasub_sketch_foldfirst_cfg,
    "sdlora_sketch_foldfirst": sdlora_sketch_foldfirst_cfg,
    "cllora_sketch_foldfirst": cllora_sketch_foldfirst_cfg,
}


def main():
    n = 0
    for method, builder in BUILDERS.items():
        for ds_name in DATASETS:
            out_dir = os.path.join(OUT_BASE, method, ds_name)
            os.makedirs(out_dir, exist_ok=True)
            for seed in SEEDS:
                cfg = builder(ds_name, seed)
                out_path = os.path.join(out_dir, "s{}.json".format(seed))
                json.dump(cfg, open(out_path, "w"), indent=2)
                n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
