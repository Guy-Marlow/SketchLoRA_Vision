"""Config generator for the CE-instrumentation smoke test (user request 2026-08-03,
CORRECTED same day: this runs on ORACLE (real task-boundary) training, NOT
bounded_memory -- an earlier version of this script wrongly used
boundary_mode="bounded_memory"/bm_budget_mb; see trainer.py's own comments for the
parallel oracle-path CE wiring this now exercises, distinct from models/
bounded_memory_mixin.py's driver):

5 methods (SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA) x 1 seed, ImageNet-R
TRUNCATED to the first 5 tasks (stop_after_tasks=5), plain per-task
(incremental_train) oracle boundaries -- 5 runs total, meant to run sequentially
on ONE GPU.

Purpose: validate the 2026-08-03 measured-CE instrumentation
(docs/ce_profiling_implementation_plan.md) on a live H200 run before trusting any
number it produces on a real campaign -- NOT a real experiment result. Per-method
hyperparameters are copied VERBATIM from scripts/gen_imagenetr_slurm_grid_configs.py's
own METHOD_CFG (same rank=10/alpha=null/batch=48/tuned_epoch=20 convention), so
this smoke test exercises the exact same per-method settings the real
imagenetr_slurm_grid campaign uses -- only the boundary/harness choice differs
(oracle here vs bounded_memory there), matching what this specific smoke test
needs to validate.

Config differences from a plain oracle imagenetr run:
  - stop_after_tasks=5: truncates the run to the first 5 of ImageNet-R's 20 tasks
    (init_cls=10/increment=10 -- 5 tasks = 50 classes). Keeps the smoke test short
    while covering multiple real task boundaries -- trainer.py's oracle loop
    writes one CE ledger record per task (unit_idx=task, nearest_latent_task=task
    exactly, no reconstruction needed the way bounded_memory's cycle-based
    records require), so 5 tasks gives 5 directly-inspectable records.
  - final_metrics=true: REQUIRED to activate both MetricsLogger and the new CE
    ledger/profiling in trainer.py's oracle loop -- both are gated behind this
    single flag (see trainer.py) so every OTHER experiment track sharing
    trainer.py (icarl/der/foster/aper_*/ranpac/...) stays completely unaffected
    by this whole exercise. Without this flag, NOTHING in this smoke test would
    produce any CE data at all.
  - ce_profile_every=1: profile EVERY task, not the production default of 25.
    This IS the validation run -- sampling would defeat the purpose of checking
    that every tagged region behaves sensibly, and at only 5 tasks the total
    profiling cost is small. NOTE (trainer.py's own caveat): oracle-mode
    profiling wraps the ENTIRE incremental_train() call (all epochs, all steps),
    not just one epoch the way bounded_memory samples -- appropriate for this
    5-task smoke test, but ce_profile_every=1 would be much more expensive on a
    full-length oracle campaign.

SEED is a single value (not swept) -- this is deliberately a single-cell smoke
test, not a grid. Change SEED below if a different seed is more useful to inspect.

Device: "device": ["0"] uniformly, matching every other SLURM-targeted config in
this project -- physical GPU selection happens via CUDA_VISIBLE_DEVICES in the
.slurm script, not here.
"""
import json
import os

OUT_DIR = "exps/ce_smoke_imagenetr5t"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 1993      # single seed -- one GPU, sequential, per user request

BASE = dict(
    dataset="imagenetr",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, scenario="cil",
    pretrained=True, print_forget=False,
    final_metrics=True,       # REQUIRED -- activates MetricsLogger + the CE ledger (see module docstring)
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    # NO boundary_mode key -- omitting it (rather than setting it to anything)
    # is what makes trainer.py fall through to the plain oracle per-task loop;
    # see trainer.py's own if/elif chain on args.get("boundary_mode").
    device=["0"],
    stop_after_tasks=5,       # smoke-test truncation -- see module docstring
    ce_profile_every=1,       # full-density profiling for validation -- see module docstring
)

# IDENTICAL to scripts/gen_imagenetr_slurm_grid_configs.py's own METHOD_CFG --
# copied, not re-derived, so this smoke test exercises the real campaign's exact
# per-method settings.
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

# order matters -- execution order in scripts/ce_smoke_imagenetr5t.slurm
METHODS = ["seqlora", "sketchlora", "olora", "inflora", "treelora"]

configs = []
for method in METHODS:
    cfg = dict(BASE)
    cfg.update(METHOD_CFG[method])
    cfg["seed"] = [SEED]
    cfg["prefix"] = "ce_smoke_imagenetr5t_{}_s{}".format(method, SEED)
    fname = "{}_s{}.json".format(method, SEED)
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    print(c)
