"""SketchLoRA-orth (models/sketchlora_align.py, sketchlora_align_mode="orth",
sketchlora_align_weight=0.5 -- O-LoRA's own orthogonality formula, applied
between the residual and the frozen sketch) on ImageNet-R-20t, 3 seeds
(1993/1996/1999), two arms:

  fixed:    svd_rank=10, svd_energy_target unset (fixed-rank sketch, no
            adaptive growth) -- mirrors exps/tsne_drift/sketchlora_
            fixedrank_cifar224_10t_s1993.json's rank-selection settings.
  adaptive: svd_energy_target=0.01, sketchlora_rank_cap=128,
            sketchlora_admission=bounded_eviction -- byte-identical to
            exps/wave1_final/sketchlora_imagenetr_s1993.json's rank-
            selection settings (the config the single-seed orth run this
            whole comparison has been tracking was copied from verbatim).

Both arms share every other hyperparameter with the wave1_final SketchLoRA
baseline (batch_size=48, init_lr=0.001, tuned_epoch=20, lora_rank=10,
weight_decay=0.0005, sketchlora_lora_wd=0.0) so the ONLY things that differ
between an arm's runs and the existing wave1_final baseline are (a) the
orth regularizer and (b) fixed-vs-adaptive rank selection.
"""
import copy
import json
import os

OUT_DIR = "exps/sketchlora_orth_seeds"
SEEDS = [1993, 1996, 1999]
BASE_SRC = "exps/wave1_final/sketchlora_imagenetr_s1993.json"


def build(seed):
    cfg = copy.deepcopy(json.load(open(BASE_SRC)))
    cfg["seed"] = [seed]
    cfg["model_name"] = "sketchlora_align"
    cfg["sketchlora_align_mode"] = "orth"
    cfg["sketchlora_align_weight"] = 0.5
    return cfg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []

    for seed in SEEDS:
        # -- fixed-rank arm: strip adaptive-rank-selection keys back to the
        # fixed-mode defaults (mirrors exps/tsne_drift/sketchlora_fixedrank_
        # cifar224_10t_s1993.json -- no svd_energy_target/rank_cap/admission).
        cfg = build(seed)
        cfg.pop("svd_energy_target", None)
        cfg.pop("sketchlora_rank_cap", None)
        cfg.pop("sketchlora_admission", None)
        cfg.pop("sketchlora_eviction_reading", None)
        cfg["prefix"] = "sketchlora_orth_seeds_fixed_imagenetr_s{}".format(seed)
        out = os.path.join(OUT_DIR, "fixed_s{}.json".format(seed))
        json.dump(cfg, open(out, "w"), indent=2)
        written.append(out)

        # -- adaptive-rank arm: byte-identical rank-selection settings to
        # wave1_final's own SketchLoRA config (already present via BASE_SRC).
        cfg = build(seed)
        cfg["prefix"] = "sketchlora_orth_seeds_adaptive_imagenetr_s{}".format(seed)
        out = os.path.join(OUT_DIR, "adaptive_s{}.json".format(seed))
        json.dump(cfg, open(out, "w"), indent=2)
        written.append(out)

    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
