"""LoRA-Subtraction-Sketch (fold-before-eval variant) STANDALONE re-run
(2026-11-14 user request) -- a direct rip of scripts/
gen_cllora_lorasub_wave1_configs.py's own import-the-settled-builder
pattern, trimmed to this one method only.

model_name="lorasub_sketch_foldfirst" is the ALREADY-EXISTING, ALREADY-
CORRECT fold-before-eval codepath (models/lorasub_sketch_foldfirst.py,
registered in utils/factory.py) from the original scripts/
gen_sketch_foldfirst_wave1_configs.py campaign -- NOT model_name=
"lorasub_sketch" (the separate, still-current-timing base method that
cllora_lorasub_wave1's own "lorasub_sketch" sub-campaign uses). These are
two distinct files/model_names on the cluster; do not confuse them.

HYPERPARAMETERS: imported directly from scripts/
gen_sketch_foldfirst_wave1_configs.py::lorasub_sketch_foldfirst_cfg --
byte-identical values (lora_rank=10, svd_rank=10, lora_n_slots=2,
merge_op="qrsvd", svd_thres=0.99, use_proto_repulsion=True,
proto_repulsion_weight=0.2, proto_margin=2.0, tuned_epoch=20,
batch_size=48, weight_decay=0.0, min_lr=0.0; init_lr per-dataset:
cifar224=1e-3, everything else=3e-3; backbone_type=
"vit_base_patch16_224_lorasub"), not re-derived.

`prefix` is given a FRESH tag (lorasub_sketch_foldfirst_wave2_<dataset>_
s<seed>), unlike cllora_sketch_foldfirst_cfg's own choice to leave its
prefix unchanged in gen_cllora_lorasub_wave1_configs.py. That precedent
relied on a provable guarantee (every prior attempt under that tag
crashed before writing a metrics file). Here the equivalent guarantee is
weaker -- the svd_sketching_vision_backup snapshot has no run_logs/final/
lorasub_sketch_foldfirst/ directory at all, but that snapshot's run_logs/
tree is empty for EVERY model, not just this one, so its absence isn't
strong evidence on its own. A fresh prefix sidesteps the question
entirely: even if scripts/sketch_foldfirst_wave1.slurm (the ORIGINAL
campaign this method's config was first built for) is ever resubmitted
independently, it cannot collide with this standalone re-run's own
metrics tag.

Only the OUTPUT location changes from the source campaign -- a fresh
exps/ dir (lorasub_sketch_foldfirst_wave2) so this standalone re-run's
configs don't sit on top of (or get confused with) either
exps/sketch_foldfirst_wave1's original 45-run bundle or
exps/cllora_lorasub_wave1's in-progress configs.

5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) = 15 runs, this
method only.
"""
import json
import os

from gen_sketch_foldfirst_wave1_configs import lorasub_sketch_foldfirst_cfg

OUT_BASE = "exps/lorasub_sketch_foldfirst_wave2"
SEEDS = [1993, 1996, 1999]
DATASETS = ["cifar224", "food101", "imagenetr", "omnibenchmark1k", "sun397"]


def _fresh_prefix_cfg(ds_name, seed):
    cfg = lorasub_sketch_foldfirst_cfg(ds_name, seed)
    cfg["prefix"] = "lorasub_sketch_foldfirst_wave2_{}_s{}".format(ds_name, seed)
    return cfg


def main():
    n = 0
    for ds_name in DATASETS:
        out_dir = os.path.join(OUT_BASE, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        for seed in SEEDS:
            cfg = _fresh_prefix_cfg(ds_name, seed)
            out_path = os.path.join(out_dir, "s{}.json".format(seed))
            json.dump(cfg, open(out_path, "w"), indent=2)
            n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
