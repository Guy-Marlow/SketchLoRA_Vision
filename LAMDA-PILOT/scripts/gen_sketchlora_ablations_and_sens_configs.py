"""SketchLoRA ablations-and-sensitivity campaign (2026-08-05 user request):
ImageNet-R full 20-task split, oracle boundaries, 3 seeds (1993/1996/1999),
rank-10/batch-48/tuned_epoch-20 convention -- same base HPs as
sketchlora_ablations_imagenetr20t. THREE sub-experiments, each its own
subfolder under exps/sketchlora_ablations_and_sens/:

1. reduce_merge_sens (6 configs): canonical (merge_op=randsvd) vs. the new
   reduce_merge ablation (models/sketchlora.py, added same session) -- same
   bounded_eviction/adaptive-threshold/eps=0.01/rank=10 settings for both,
   ONLY merge_op differs, isolating that one axis. NO RANK CAP for both
   (explicit user request for this campaign -- differs from the existing
   sketchlora_ablations_imagenetr20t "current" config, which uses rank_cap=128;
   not a mismatch, a deliberate choice for this comparison).

2. period_sens (9 configs): svd_period in {2, 5, 10}, lora_n_slots set to
   period+1 for each (a REQUIRED pairing -- svd_period needs that many
   residual slots actually allocated, models/sketchlora.py:119's own
   docstring; period-2 with the default lora_n_slots=2 would silently try to
   route to a slot that was never allocated). Verified against the actual
   model code before writing this (not assumed): _train_adapter()/
   _stream_slot() route via `RESIDUAL + (task % svd_period)`, fold only
   fires at `(task+1) % svd_period == 0` (or the forced final-task fold for
   a trailing partial period), and the adaptive-threshold/bounded_eviction
   rank-selection path already generalizes over `_residual_slots()` (P
   slots) with no P-specific code needed -- confirmed correct via a real
   5-task local smoke test at period=2 before this generator was written
   (rank grew 16.2 -> 29.8 -> 33.9 with rank_cap unset, sketch_diag logged
   only at the period boundaries, exactly as expected).

3. epsilon_sens (30 configs): svd_energy_target in
   {0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.005, 0.0}. 0.01 is
   deliberately EXCLUDED -- that data already exists
   (sketchlora_ablations_imagenetr20t's "current" variant). eps=0.0 is
   included because a local smoke test (3-task ImageNet-R, 1 epoch) confirmed
   it runs without error before this generator was written -- with eps=0.0,
   `_compress()`'s k_eps computation clips to "keep essentially everything"
   (not a crash, not an off-by-one -- 1.0 - eps = 1.0, so cum < 1.0 selects
   every direction below the numerically-exact top singular value).

Every config: dataset=imagenetr, init_cls=10, increment=10 (full 20-task
split, no stop_after_tasks -- this is a real evaluation campaign, not a
truncated sweep), rank=10, batch=48, tuned_epoch=20, lora_alpha=null,
sketchlora_admission=bounded_eviction, sketchlora_rank_cap=null (no cap,
every sub-experiment), final_metrics=true/print_forget=true (full logging --
unlike the LR sweep, this campaign needs persistent memory/CE/top1/top5/
sketch_diag data, not just a fast accuracy check).
"""
import json
import os

OUT_BASE = "exps/sketchlora_ablations_and_sens"
RUN_LOGS_BASE = "run_logs/sketchlora_ablations_and_sens"
SEEDS = [1993, 1996, 1999]

BASE = dict(
    dataset="imagenetr",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora", backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    lora_merge=True, lora_train_merge=True,
    svd_rank=10, svd_oversampling=10, sketch_diag=True,
    sketchlora_admission="bounded_eviction", sketchlora_rank_cap=None,
    sketchlora_lora_wd=0.0, sketchlora_eviction_reading="conformant",
    device=["0"],
)


def write_cfg(subdir, variant, seed, overrides):
    cfg = dict(BASE)
    cfg["seed"] = [seed]
    cfg.update(overrides)
    cfg["prefix"] = "sketchlora_ablations_and_sens_{}_{}_s{}".format(subdir, variant, seed)
    # 2026-08-05: sketch_diag's reconstruction-error JSON now writes DIRECTLY
    # into this campaign's own <subdir>/diag/ (models/sketchlora.py's
    # sketchlora_diag_dir opt-in) instead of the shared run_logs/ location
    # every other campaign (including wave1_final, running concurrently on
    # the cluster right now) defaults to -- avoids a real cross-campaign
    # collision, since wave1_final's SketchLoRA/ImageNet-R cells use the same
    # (dataset, eps=0.01, split) that would otherwise produce the identical
    # shared filename. No move step needed afterward; this IS the final
    # location.
    # forward slashes explicit, NOT os.path.join -- this runs on the Windows
    # dev machine but the resulting path string is read back by main.py on
    # the (Linux) cluster, where a literal backslash baked in by Windows'
    # os.path.join is just another filename character, not a separator.
    cfg["sketchlora_diag_dir"] = "{}/{}/diag".format(RUN_LOGS_BASE, subdir)
    outdir = os.path.join(OUT_BASE, subdir)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "{}_s{}.json".format(variant, seed))
    json.dump(cfg, open(path, "w"), indent=2)
    return path


def main():
    written = []

    # 1. reduce_merge_sens
    for variant, merge_op in [("canonical", "randsvd"), ("reduce_merge", "reduce_merge")]:
        for seed in SEEDS:
            written.append(write_cfg("reduce_merge_sens", variant, seed, dict(
                merge_op=merge_op, svd_energy_target=0.01,
                lora_n_slots=2,
            )))

    # 2. period_sens
    for period in (2, 5, 10):
        variant = "period{}".format(period)
        for seed in SEEDS:
            written.append(write_cfg("period_sens", variant, seed, dict(
                merge_op="randsvd", svd_energy_target=0.01,
                svd_period=period, lora_n_slots=period + 1,
            )))

    # 3. epsilon_sens
    for eps in (0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.005, 0.0):
        variant = "eps{}".format(eps)
        for seed in SEEDS:
            written.append(write_cfg("epsilon_sens", variant, seed, dict(
                merge_op="randsvd", svd_energy_target=eps,
                lora_n_slots=2,
            )))

    print("wrote {} configs under {}/".format(len(written), OUT_BASE))


if __name__ == "__main__":
    main()
