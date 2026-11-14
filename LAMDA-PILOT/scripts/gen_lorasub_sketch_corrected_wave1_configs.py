"""LoRA-Subtraction-Sketch STANDALONE re-run (2026-11-14 user request) --
the fold-timing fix to models/lorasub_sketch.py (merged 2026-11-13 from the
former models/lorasub_sketch_foldfirst.py: the QR-SVD fold now runs at the
END of each task's own incremental_train, right after _train() returns,
instead of at the START of the next task) was made AFTER the
cllora_lorasub_wave1 campaign's own lorasub_sketch sub-campaign had already
started executing on the cluster with the pre-fix code. This generator
produces a completely independent, correctly-fixed set of configs so the
fix can be validated without waiting for (or colliding with)
cllora_lorasub_wave1's own eventual lorasub_sketch runs.

HYPERPARAMETERS: byte-identical to the already-settled
scripts/gen_lorasub_sdlora_sketch_wave1_configs.py::lorasub_sketch_cfg
(lora_rank=10, svd_rank=10, lora_n_slots=2, merge_op="qrsvd",
svd_thres=0.99, use_proto_repulsion=True, proto_repulsion_weight=0.2,
proto_margin=2.0, tuned_epoch=20, batch_size=48, weight_decay=0.0,
min_lr=0.0; init_lr per-dataset: cifar224=1e-3, everything else=3e-3) --
not re-derived, same convention every campaign generator in this project
follows.

`prefix` is a FRESH tag (lorasub_sketch_corrected_wave1_*), distinct from
BOTH lorasub_sdlora_sketch_wave1 (stale -- predates the DRS-probe/init/
prototype fix to models/lorasub.py, commit 8800724) AND
cllora_lorasub_wave1 (stale -- predates THIS fold-timing fix). Reusing
either would let scripts/_bankcap_run_done.py's resumability check skip
these as already-done and silently report stale numbers as if they
validated the fixed code -- exactly the bug scripts/
gen_cllora_lorasub_wave1_configs.py's own prefix-override already guards
against for the OTHER stale source, repeated here for this one.

5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) = 15 runs, this method
only -- CL-LoRA-Sketch-FoldFirst and LoRA-Subtraction are unaffected by
this fix and continue under cllora_lorasub_wave1 as before.
"""
import json
import os

OUT_BASE = "exps/lorasub_sketch_corrected_wave1"
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
        prefix="lorasub_sketch_corrected_wave1_{}_s{}".format(ds_name, seed),
    )
    return cfg


def main():
    n = 0
    for ds_name in DATASETS:
        out_dir = os.path.join(OUT_BASE, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        for seed in SEEDS:
            cfg = lorasub_sketch_cfg(ds_name, seed)
            out_path = os.path.join(out_dir, "s{}.json".format(seed))
            json.dump(cfg, open(out_path, "w"), indent=2)
            n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
