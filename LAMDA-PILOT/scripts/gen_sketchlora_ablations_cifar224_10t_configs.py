"""SketchLoRA ablation series, ported to CIFAR-100-10t (user request
2026-08-13). Reuses the exact ImageNet-R-20t ablation configs this project
already ran (originally split across two SLURM scripts -- scripts/
sketchlora_ablations_imagenetr20t.slurm/gen_sketchlora_ablations_imagenetr20t_
configs.py for the SVD-variant/countsketch/noadapt cells, plus
exps/sketchlora_ablations_and_sens/reduce_merge_sens/ for the reduce_merge
cell run separately) as the literal SOURCE OF TRUTH: each config below is
loaded from its ImageNet-R counterpart and only DATASET-DEPENDENT fields are
patched (dataset, prefix, and reduce_merge's sketchlora_diag_dir) -- every
other field (merge_op, admission policy, rank_cap, svd_energy_target,
optimizer/schedule, epochs, batch size, etc.) is copied byte-for-byte, so
there is no risk of a hand-transcribed METHOD_CFG dict silently dropping or
altering a field relative to what actually ran on ImageNet-R.

init_cls/increment are UNCHANGED at 10/10 -- CIFAR-100 (100 classes) and
ImageNet-R (200 classes) both split into the same 10/10 shape here, just
10 tasks instead of 20 (CIFAR-100-10t, per this project's standing wave1_final
convention for that dataset).

SIX variants (matching plot_ablation_curves.py's Set-1 labels), excluding
"Random SVD + Floor" (sketchlora_current, the baseline) and "Sequential LoRA"
(seqlora) -- both already have completed CIFAR-100 data from other campaigns,
per explicit user instruction:
  exactsvd     -- "Exact SVD + Floor"        (merge_op=exactsvd, bounded_eviction)
  globaleps    -- "Random SVD"               (merge_op=randsvd, global_eps, no floor)
  reduce_merge -- "Reversed Sketch/Merge"    (merge_op=reduce_merge)
  fixedrank    -- "Fixed + Random SVD"       (merge_op=randsvd, no svd_energy_target)
  countsketch  -- "Fixed + CountSketch"      (merge_op=countsketch, no svd_energy_target)
  noadapt      -- "No Adaptation"            (frozen backbone, no LoRA)

3 seeds (1993/1996/1999), matching every other campaign in this project.
"""
import copy
import json
import os

OUT_DIR = "exps/sketchlora_ablations_cifar224_10t"
SEEDS = [1993, 1996, 1999]
DATASET = "cifar224"

# output_basename -> (source_path_template, short_tag_for_prefix). short_tag
# matches the ImageNet-R source's OWN prefix convention exactly (e.g.
# "sketchlora_ablations_imagenetr20t_exactsvd_s1993", NOT
# "..._sketchlora_exactsvd_s1993") -- kept distinct from output_basename
# (which matches the ImageNet-R source's FILENAME convention, e.g.
# sketchlora_exactsvd_s<seed>.json) so the generated prefix field stays
# predictable for the SLURM script's own skip-check.
SOURCES = {
    "sketchlora_exactsvd": (
        "exps/sketchlora_ablations_imagenetr20t/sketchlora_exactsvd_s{seed}.json", "exactsvd"),
    "sketchlora_globaleps": (
        "exps/sketchlora_ablations_imagenetr20t/sketchlora_globaleps_s{seed}.json", "globaleps"),
    "sketchlora_fixedrank": (
        "exps/sketchlora_ablations_imagenetr20t/sketchlora_fixedrank_s{seed}.json", "fixedrank"),
    "sketchlora_countsketch": (
        "exps/sketchlora_ablations_imagenetr20t/sketchlora_countsketch_s{seed}.json", "countsketch"),
    "noadapt": (
        "exps/sketchlora_ablations_imagenetr20t/noadapt_s{seed}.json", "noadapt"),
    "reduce_merge": (
        "exps/sketchlora_ablations_and_sens/reduce_merge_sens/reduce_merge_s{seed}.json", "reduce_merge"),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for variant, (src_template, short_tag) in SOURCES.items():
        for seed in SEEDS:
            src_path = src_template.format(seed=seed)
            cfg = copy.deepcopy(json.load(open(src_path)))

            cfg["dataset"] = DATASET
            cfg["prefix"] = "sketchlora_ablations_cifar224_10t_{}_s{}".format(short_tag, seed)
            if variant == "reduce_merge":
                cfg["sketchlora_diag_dir"] = "run_logs/sketchlora_ablations_cifar224_10t/reduce_merge/diag"

            out_path = os.path.join(OUT_DIR, "{}_s{}.json".format(variant, seed))
            json.dump(cfg, open(out_path, "w"), indent=2)
            written.append(out_path)
    print("wrote {} configs to {}/ (from {} variants x {} seeds)".format(
        len(written), OUT_DIR, len(SOURCES), len(SEEDS)))


if __name__ == "__main__":
    main()
