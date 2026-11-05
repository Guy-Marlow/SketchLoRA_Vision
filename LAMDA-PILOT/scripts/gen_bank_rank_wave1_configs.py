"""Bank-effective-rank comparison campaign (2026-11-06 user request):
InfLoRA-Bank, O-LoRA, and LoRA-Subtraction (growing-bank variant), each on
its own vanilla/settled hyperparameters, across all 5 master-table datasets
(CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t, SUN397-10t,
Food-101-20t), 3 seeds each (1993/1996/1999) -- 3 methods x 5 datasets x 3
seeds = 45 runs. Every config also sets log_bank_rank=true (utils/
rank_diagnostics.py, added this session) -- the summed-bank effective-rank
diagnostic (singular values needed for 99% spectral energy of the
cumulative per-task adapter update), the whole point of this campaign:
comparing how high that rank climbs across methods with different (or no)
cross-task interference-avoidance mechanisms. Verified via a smoke test on
all three methods before this generator was written (see the conversation
record) -- InfLoRA-Bank and O-LoRA both climb close to their own linear
ceiling, LoRA-Subtraction stays almost flat near its single-task rank.

HYPERPARAMETERS, per method:

InfLoRA-Bank (model_name=inflora_bank): identical to the existing
exps/wave1_final/inflora_<dataset>_s1993.json configs (the master-table
source for plain InfLoRA -- model_name swapped, everything else byte-
identical, since InfLoRA-Bank is mathematically identical to plain InfLoRA
except where a trained task's B lives afterward -- see models/
inflora_bank.py's own module docstring) -- lora_rank=10, lamb=0.95,
lame=1.0, tuned_epoch=20, batch_size=48, weight_decay=0.0005, min_lr=0.0,
lora_merge=true. init_lr is per-dataset: 0.0005 everywhere except
imagenetr (0.001) -- read directly off those committed configs, not
re-derived.

O-LoRA (model_name=olora): identical to the existing exps/
wave1_final_completion/olora_<dataset>_s1993.json configs -- init_lr=0.001
uniformly across all 5 datasets, lamda_1=0.5 (orthogonality weight),
lamda_2=0.0, lora_rank=10, same epoch/batch/wd/min_lr, lora_merge=true
(the bank-accumulating default -- see the conversation record's direct
confirmation that every existing O-LoRA config already runs with
accumulation on).

LoRA-Subtraction (model_name=lorasub, growing-bank variant): this project's
own just-settled per-dataset LRs from this session's adaptive LR sweep
(exps/lorasub_lr_tune/SWEEP_LOG.md) -- cifar224=1e-3, imagenetr=3e-3,
omnibenchmark1k=3e-3, sun397=3e-3, food101=3e-3 (no prior "wave1_final"
convention exists for this method -- it postdates that campaign). FULL
task counts (no stop_after_tasks, unlike the truncated LR-tuning runs),
tuned_epoch=20, batch_size=48, weight_decay=0.0 (matches this project's
own established LoRA-Subtraction convention, not the 0.0005 the other two
methods use), lora_rank=10, svd_thres=0.99, use_proto_repulsion=true (the
catastrophic-collapse fix found and validated earlier this session --
REQUIRED, not optional, for this method to produce usable CIL numbers),
proto_repulsion_weight=0.2, proto_margin=2.0.

Per-dataset splits (init_cls/increment), identical across all three
methods, read directly off the existing committed wave1_final/
wave1_final_completion configs: cifar224=10/10 (10 tasks), imagenetr=10/10
(20 tasks), omnibenchmark1k=10/10 (100 tasks), sun397=37/40 (10 tasks),
food101=6/5 (20 tasks).

bank_rank_energy_target left at its 0.99 default (explicit anyway, for
clarity) throughout.
"""
import json
import os

OUT_BASE = "exps/bank_rank_wave1"
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
    device=["0"], log_bank_rank=True, bank_rank_energy_target=0.99,
)

INFLORA_BANK_LR = {
    "cifar224": 0.0005, "imagenetr": 0.001, "omnibenchmark1k": 0.0005,
    "sun397": 0.0005, "food101": 0.0005,
}

LORASUB_LR = {
    "cifar224": 0.001, "imagenetr": 0.003, "omnibenchmark1k": 0.003,
    "sun397": 0.003, "food101": 0.003,
}


def inflora_bank_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0005,
        model_name="inflora_bank", backbone_type="vit_base_patch16_224_lora",
        lora_merge=True, lamb=0.95, lame=1.0,
        init_lr=INFLORA_BANK_LR[ds_name],
        prefix="bank_rank_wave1_inflora_bank_{}_s{}".format(ds_name, seed),
    )
    return cfg


def olora_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0005,
        model_name="olora", backbone_type="vit_base_patch16_224_lora",
        lora_merge=True, lamda_1=0.5, lamda_2=0.0,
        init_lr=0.001,
        prefix="bank_rank_wave1_olora_{}_s{}".format(ds_name, seed),
    )
    return cfg


def lorasub_cfg(ds_name, seed):
    cfg = dict(COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], weight_decay=0.0,
        model_name="lorasub", backbone_type="vit_base_patch16_224_lorasub",
        svd_thres=0.99, use_proto_repulsion=True,
        proto_repulsion_weight=0.2, proto_margin=2.0,
        init_lr=LORASUB_LR[ds_name],
        prefix="bank_rank_wave1_lorasub_{}_s{}".format(ds_name, seed),
    )
    return cfg


BUILDERS = {
    "inflora_bank": inflora_bank_cfg,
    "olora": olora_cfg,
    "lorasub": lorasub_cfg,
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
