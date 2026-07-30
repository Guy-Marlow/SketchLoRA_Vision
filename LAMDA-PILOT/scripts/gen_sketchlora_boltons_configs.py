"""Config generator for the SketchLoRA bolt-on factorial (impl_plan_7.27.2026
Part 1 sec 1.4), user-narrowed scope (2026-07-27): 15-task OmniBenchmark-1K
(not the plan's own 30-task convention) x {50, 100}MB flat budgets (not the
plan's own 0.2x/0.5x mean-task-size convention) x the plan's own 6 bolt-on
arms, single seed 1993 (matches the existing round2_anchor grid for direct
comparability -- the plan's "off" arm at 50MB IS that already-completed run,
reused rather than rerun; see gen script output).

Arms (plan sec 1.4): off, FD, LM-plateau, CA, FD+LM, FD+LM+CA. "LM-period" is
NOT in the plan's own run matrix -- period mode exists in the code (models/
sketchlora.py) for config-surface completeness but is not exercised by this
factorial.
"""
import json
import os

OUT_DIR = "exps/sketchlora_boltons"
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
    stop_after_tasks=15,
    device=["PLACEHOLDER"],
    model_name="sketchlora", lora_merge=True, lora_train_merge=True,
    svd_rank=10, svd_oversampling=10, svd_energy_target=0.01,
    lora_n_slots=2, sketch_diag=True, merge_op="randsvd",
    sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
    sketchlora_lora_wd=0.0, sketchlora_eviction_reading="conformant",
    init_lr=0.001,
)

ARMS = {
    "off": dict(),
    "fd": dict(fd_shrinkage=True),
    "lmplateau": dict(lazy_merge="plateau"),
    "ca": dict(classifier_alignment=True, ca_steps=300, ca_batch=128, ca_lr=1e-3),
    "fd_lm": dict(fd_shrinkage=True, lazy_merge="plateau"),
    "fd_lm_ca": dict(fd_shrinkage=True, lazy_merge="plateau",
                      classifier_alignment=True, ca_steps=300, ca_batch=128, ca_lr=1e-3),
}
BUDGETS = [50, 100]

configs = []
skipped = []
for budget in BUDGETS:
    for arm_name, arm_cfg in ARMS.items():
        cfg = dict(BASE)
        cfg.update(arm_cfg)
        cfg["bm_budget_mb"] = budget
        cfg["prefix"] = "sketchlora_boltons_omni15t_{}mb_{}".format(budget, arm_name)
        fname = "{}mb_{}.json".format(budget, arm_name)
        if budget == 50 and arm_name == "off":
            # already run (exps/round2_anchor/sketchlora_50mb_15t.json, seed 1993,
            # partial:false, result in run_logs/boundedmem_sketchlora_round2_anchor_
            # omni15t_50mb_sketchlora_s1993.json) -- identical config modulo prefix;
            # write it anyway for record-keeping but DO NOT queue it for a run.
            skipped.append(fname)
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    tag = " (SKIP -- reuse existing round2_anchor result)" if c in skipped else ""
    print(c + tag)
