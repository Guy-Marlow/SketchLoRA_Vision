"""Config generator for the sketchlora_ablations_imagenetr20t REPAIR/EXTENSION
round (user request 2026-08-04, following analysis of job 22181's results,
see sketchlora_ablations_imagenetr20t.md). Writes into the SAME directory as
scripts/gen_sketchlora_ablations_imagenetr20t_configs.py
(exps/sketchlora_ablations_imagenetr20t/) but does NOT touch that script or
regenerate its 18 existing configs -- this only ADDS new files. Four things:

  1. seqlora        -- SeqLoRA on the full 20-task split, for comparison
                        against the SketchLoRA family. Hyperparameters copied
                        verbatim from scripts/gen_imagenetr_slurm_grid_configs.py's
                        own METHOD_CFG (model_name=seqlora, lora_merge=False,
                        init_lr=0.0003) -- the project's one standing SeqLoRA
                        convention, not re-derived.
  2. noadapt         -- the no-adaptation baseline (models/noadapt.py, added
                        alongside this script): frozen ViT backbone, zero
                        gradient steps, NCM prototype head. lora_n_slots=1
                        (only slot 0 is ever touched, and even that only to
                        satisfy freeze_to_task's bookkeeping -- never trained)
                        -- deliberately smaller than the other configs'
                        lora_n_slots=2, an honest reflection of what this
                        method actually allocates. tuned_epoch left at 1
                        (inert: models/noadapt.py::_train never loops epochs
                        at all, no optimizer is ever constructed) rather than
                        the usual 20, so the config doesn't imply gradient
                        training that never happens.
  3. fixedrank_ca    -- fixedrank (svd_energy_target omitted -> rank pinned at
                        svd_rank=lora_rank=10, admission=global_eps,
                        merge_op=randsvd, rank_cap=128 -- identical to the
                        existing "fixedrank" variant) PLUS
                        classifier_alignment=True (ca_steps=300/ca_batch=128/
                        ca_lr=0.001, copied verbatim from exactsvd_ca's own
                        CA hyperparameters for comparability). Isolates: with
                        the merge algorithm and rank policy held fixed at
                        fixedrank's settings, how much of fixedrank's ~3.6pt
                        deficit vs. "current" (see sketchlora_ablations_
                        imagenetr20t.md) is closable by correcting classifier
                        HEAD drift alone, vs. requiring the adaptive rank
                        itself.
  4. fixedrank_exactsvd_ca -- fixedrank_ca but merge_op=exactsvd instead of
                        randsvd (i.e. current's exactsvd axis change ALSO
                        applied on top of fixedrank+CA). Isolates the same
                        question as (3) but with the merge-approximation axis
                        held at its best (exact) setting too, to see how much
                        FURTHER ground exact-SVD reconstruction recovers once
                        CA has already been given credit -- potentially at
                        some accuracy cost from the fixed (non-adaptive) rank
                        that exact reconstruction alone cannot fix.

All four use this campaign's own BASE dict verbatim (ImageNet-R full 20-task
split, oracle boundaries -- no boundary_mode key, final_metrics=True,
ce_profile_every=0), i.e. identical harness conventions to every other
sketchlora_ablations_imagenetr20t cell, so results are directly comparable
without a separate normalization step.

NOTE this script does NOT regenerate exactsvd_ca's own config -- that variant's
config is unchanged (the bug fixed in models/sketchlora.py 2026-08-05 was a
CODE gap, not a config problem; exactsvd_ca's json from the original
generator is already correct and is reused as-is). The repair round's SLURM
script (scripts/sketchlora_ablations_imagenetr20t_repair.slurm) force-reruns
that existing config to overwrite its previously-corrupted output.
"""
import json
import os

OUT_DIR = "exps/sketchlora_ablations_imagenetr20t"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(
    dataset="imagenetr",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, scenario="cil",
    pretrained=True, print_forget=False, final_metrics=True,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    device=["0"],
    ce_profile_every=0,
    # no boundary_mode key -- ordinary per-task oracle loop, real task
    # boundaries (identical to the rest of this campaign).
)

SEEDS = [1993, 1996, 1999]

CA_HP = dict(classifier_alignment=True, ca_steps=300, ca_batch=128, ca_lr=0.001)

EXTRA_CFG = {
    "seqlora": dict(model_name="seqlora", lora_merge=False, init_lr=0.0003),
    "noadapt": dict(
        model_name="noadapt", lora_merge=False, lora_n_slots=1,
        tuned_epoch=1,   # inert -- models/noadapt.py::_train never runs an epoch loop
        init_lr=0.0,     # inert -- no optimizer is ever constructed
    ),
    "fixedrank_ca": dict(
        model_name="sketchlora", lora_merge=True, lora_train_merge=True,
        svd_rank=10, svd_oversampling=10,
        lora_n_slots=2, sketch_diag=True,
        sketchlora_lora_wd=0.0, init_lr=0.001,
        merge_op="randsvd",
        # svd_energy_target intentionally absent -> fixed-rank path (matches "fixedrank")
        sketchlora_admission="global_eps",
        sketchlora_rank_cap=128,
        **CA_HP,
    ),
    "fixedrank_exactsvd_ca": dict(
        model_name="sketchlora", lora_merge=True, lora_train_merge=True,
        svd_rank=10, svd_oversampling=10,
        lora_n_slots=2, sketch_diag=True,
        sketchlora_lora_wd=0.0, init_lr=0.001,
        merge_op="exactsvd",
        sketchlora_admission="global_eps",
        sketchlora_rank_cap=128,
        **CA_HP,
    ),
}

# order matters -- execution order in the repair .slurm script
VARIANTS = ["seqlora", "noadapt", "fixedrank_ca", "fixedrank_exactsvd_ca"]

configs = []
for variant in VARIANTS:
    for seed in SEEDS:
        cfg = dict(BASE)
        cfg.update(EXTRA_CFG[variant])
        cfg["seed"] = [seed]
        cfg["prefix"] = "sketchlora_ablations_imagenetr20t_{}_s{}".format(variant, seed)
        fname = "{}_s{}.json".format(variant, seed)
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    print(c)
