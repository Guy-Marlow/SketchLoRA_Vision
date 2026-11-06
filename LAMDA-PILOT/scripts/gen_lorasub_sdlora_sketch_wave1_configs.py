"""LoRA-Subtraction-Sketch + SD-LoRA-Sketch oracle campaign (2026-11-06 user
request): both fixed-size SketchLoRA-family extensions, across all 5
master-table datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
SUN397-10t, Food-101-20t), 3 seeds each (1993/1996/1999) -- 2 methods x 5
datasets x 3 seeds = 30 runs.

Both methods were audited (code read end-to-end against the backbone's
_lora_delta/freeze_to_task/forward path) and smoke/real-run-validated
locally before this generator was written -- see the conversation record:
  - LoRA-Subtraction-Sketch: already running 3 full seeds on ImageNet-R
    (exps/lorasub_imagenetr_full/{bank,sketch}_s*.json) as of this
    campaign's construction -- sketch_s1999 was mid-run and left untouched.
    Confirmed fixed 2-slot (SKETCH+RESIDUAL) bank (no add_task_slot() call
    anywhere in models/lorasub_sketch.py), confirmed the new RESIDUAL
    adapter trains via a merge=True forward that sums frozen SKETCH +
    live RESIDUAL on top of the frozen k_proj/v_proj base (backbone/
    vit_lorasub.py::Attention_LoRASub._lora_delta's merge branch), and
    confirmed CIL eval (bare net(x), default_task=RESIDUAL, network-level
    merge=True) also sums both slots -- _eval_adapter=SKETCH only affects
    the separate TIL/_forward_task path, not CIL. The one real crash risk
    found: models/lorasub_sketch.py's first-fold transplant asserts
    svd_rank == lora_rank exactly (an exact, lossless first copy) -- both
    set to 10 everywhere below, so this never fires.
  - SD-LoRA-Sketch: smoke-tested on CIFAR-100-10t this session (3-task/
    10-epoch, then a 4-task/2-epoch integration smoke, then a 6-task/
    20-epoch run reaching task 4 before being stopped for this campaign).
    Confirmed fixed-size bank (VisionTransformerSDLoRASketch.add_task_slot
    raises NotImplementedError; n_tasks forced to 1, slot 0 reused forever)
    and confirmed the new adapter trains with respect to the frozen base
    AND the sketch (Attention_SDLoRASketch._lora_delta always returns
    sketch_term(x) + new_adapter_term(x), added to the frozen q_proj/
    v_proj output in backbone/vit_lora.py::Attention_LoRA.forward,
    identically at train and eval time -- no separate merge=False/True
    branching for this method). No dataset-specific LR tune exists yet FOR
    THE SKETCH VARIANT ITSELF -- per instruction, this campaign instead
    reuses vanilla SD-LoRA's own already-settled per-dataset LRs (exps/
    slao_sdlora_wave1_datasets/sdlora/<dataset>_s1993.json), same borrowing
    convention LoRA-Subtraction-Sketch already gets from plain
    LoRA-Subtraction below. Flagged, not hidden: those LRs were tuned for
    vanilla SD-LoRA's single global per-task magnitude scalar, not this
    method's per-projection sketch+rescale+magnitude design -- if any
    dataset's accuracy looks off, a fresh LR tune for the sketch variant
    itself is the first thing to try, not a code bug.

HYPERPARAMETERS, per method:

LoRA-Subtraction-Sketch (model_name=lorasub_sketch): identical to the
existing exps/lorasub_imagenetr_full/sketch_s1993.json / exps/
bank_rank_wave1/lorasub/<dataset>/s1993.json conventions -- lora_rank=10,
svd_rank=10, lora_n_slots=2, merge_op="qrsvd", svd_thres=0.99,
use_proto_repulsion=True, proto_repulsion_weight=0.2, proto_margin=2.0,
tuned_epoch=20, batch_size=48, weight_decay=0.0, min_lr=0.0. init_lr is
per-dataset, read directly off the committed bank_rank_wave1/lorasub
configs: cifar224=1e-3, everything else=3e-3.

SD-LoRA-Sketch (model_name=sdlora_sketch): vanilla SD-LoRA's own settled
per-dataset LRs (exps/slao_sdlora_wave1_datasets/sdlora/*.json), per
instruction -- cifar224=5e-3, food101=5e-3, sun397=5e-3,
omnibenchmark1k=5e-3, imagenetr=8e-3 (separately tuned there, not a fresh
tune or an oversight). weight_decay=5e-4 (also matches vanilla SD-LoRA's
own convention). Everything else is this session's own validated setting
for the sketch mechanism itself (no vanilla-SD-LoRA equivalent exists):
lora_rank=10, svd_rank=10, sdsketch_reg_strength=0.1, sdsketch_reg_power=1.0,
sdsketch_freeze_rescale=False, tuned_epoch=20, batch_size=48, min_lr=0.0.

Per-dataset splits (init_cls/increment), identical across both methods and
to every other wave1-style campaign in this project: cifar224=10/10 (10
tasks), imagenetr=10/10 (20 tasks), omnibenchmark1k=10/10 (100 tasks),
sun397=37/40 (10 tasks), food101=6/5 (20 tasks).
"""
import json
import os

OUT_BASE = "exps/lorasub_sdlora_sketch_wave1"
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
    # Vanilla SD-LoRA's own settled per-dataset LRs, borrowed per instruction
    # -- see module docstring.
    "cifar224": 0.005, "imagenetr": 0.008, "omnibenchmark1k": 0.005,
    "sun397": 0.005, "food101": 0.005,
}


def lorasub_sketch_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0,
        model_name="lorasub_sketch", backbone_type="vit_base_patch16_224_lorasub",
        lora_n_slots=2, svd_rank=10, merge_op="qrsvd",
        svd_thres=0.99, use_proto_repulsion=True,
        proto_repulsion_weight=0.2, proto_margin=2.0,
        init_lr=LORASUB_SKETCH_LR[ds_name],
        prefix="lorasub_sdlora_sketch_wave1_lorasub_sketch_{}_s{}".format(ds_name, seed),
    )
    return cfg


def sdlora_sketch_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0005,
        model_name="sdlora_sketch", backbone_type="vit_base_patch16_224_sdlora_sketch",
        svd_rank=10, sdsketch_reg_strength=0.1, sdsketch_reg_power=1.0,
        sdsketch_freeze_rescale=False,
        init_lr=SDLORA_SKETCH_LR[ds_name],
        prefix="lorasub_sdlora_sketch_wave1_sdlora_sketch_{}_s{}".format(ds_name, seed),
    )
    return cfg


BUILDERS = {
    "lorasub_sketch": lorasub_sketch_cfg,
    "sdlora_sketch": sdlora_sketch_cfg,
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
