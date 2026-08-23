"""Four SketchLoRA-orth + "retain" admission-rule configs, ImageNet-R-20t,
seed 1993, 2026-08-20 user design:

  retain_up / retain_down: sketchlora_admission="retain",
    sketchlora_retain_anneal="cosine", retention cosine-annealed 50%<->90%
    over the 20 tasks (up: 0.5->0.9, down: 0.9->0.5). Constant
    sketchlora_align_weight=0.5 (orth mode, O-LoRA's own formula, applied
    between residual and sketch).

  retain_up_linked / retain_down_linked: SAME retain schedule (up/down
    respectively), but sketchlora_align_weight_mode="retain_linked" --
    align_weight tracks the run's OWN instantaneous retention value that
    task via linear interpolation over [0.5,0.9] -> [0.1,0.5]. In the "down"
    run this makes align_weight mirror-DECREASE from 0.5->0.1 (tracks
    retention's actual value, not an independent task-indexed ramp) -- user-
    confirmed reading, not the alternative (weight always ramping 0.1->0.5
    by task index regardless of retention's own direction).

All four share every other hyperparameter with exps/wave1_final/
sketchlora_imagenetr_s1993.json (batch_size=48, init_lr=0.001,
tuned_epoch=20, lora_rank=10, weight_decay=0.0005, sketchlora_lora_wd=0.0).
svd_energy_target stays set (0.01) only because sketchlora_admission="retain"
requires SOME non-None energy_target at __init__ time -- its value is unused
once retain_anneal="cosine" is set (models/sketchlora.py::_retain_epsilon
branches to the annealed value first).
"""
import copy
import json
import os

OUT_DIR = "exps/sketchlora_retain_anneal"
SEED = 1993
BASE_SRC = "exps/wave1_final/sketchlora_imagenetr_s1993.json"

ARMS = {
    "retain_up": {
        "sketchlora_retain_start": 0.5, "sketchlora_retain_end": 0.9,
        "sketchlora_align_weight_mode": "constant", "sketchlora_align_weight": 0.5,
    },
    "retain_down": {
        "sketchlora_retain_start": 0.9, "sketchlora_retain_end": 0.5,
        "sketchlora_align_weight_mode": "constant", "sketchlora_align_weight": 0.5,
    },
    "retain_up_linked": {
        "sketchlora_retain_start": 0.5, "sketchlora_retain_end": 0.9,
        "sketchlora_align_weight_mode": "retain_linked",
        "sketchlora_align_weight_min": 0.1, "sketchlora_align_weight_max": 0.5,
    },
    "retain_down_linked": {
        "sketchlora_retain_start": 0.9, "sketchlora_retain_end": 0.5,
        "sketchlora_align_weight_mode": "retain_linked",
        "sketchlora_align_weight_min": 0.1, "sketchlora_align_weight_max": 0.5,
    },
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for arm, overrides in ARMS.items():
        cfg = copy.deepcopy(json.load(open(BASE_SRC)))
        cfg["seed"] = [SEED]
        cfg["model_name"] = "sketchlora_align"
        cfg["sketchlora_align_mode"] = "orth"
        cfg["sketchlora_admission"] = "retain"
        cfg["sketchlora_retain_anneal"] = "cosine"
        cfg.update(overrides)
        cfg["prefix"] = "sketchlora_retain_anneal_{}_imagenetr_s{}".format(arm, SEED)
        out = os.path.join(OUT_DIR, "{}_s{}.json".format(arm, SEED))
        json.dump(cfg, open(out, "w"), indent=2)
        written.append(out)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
