"""LR refinement sweep for CL-LoRA/EASE/TUNA/RainbowPrompt (follow-up to the
original cllora_ease_tuna_rainbow_sens grid, scripts/gen_lr_sweep_configs.py) --
explicit user-specified LR values per method, narrowed to the dataset subset
each method's own earlier sweep flagged as unsettled (see MEMORY.md's
learning-rate table: asterisked cells were already at a local peak, arrow
cells were still improving at the edge of the multiplier range and are what
this sweep re-probes).

Method/architecture hyperparameters (optimizer, scheduler, batch size, epochs,
rank/ffn-dim, etc.) are copied VERBATIM from scripts/gen_lr_sweep_configs.py's
METHOD_CFG -- nothing about the training loop changes here, only which LR
values get tried, on which datasets, and the task-count truncation (explicit
user counts this time, different from the original sweep's own truncation).

Single seed (1993), same as the original sweep. final_metrics/print_forget
both off, same rationale as before (a short LR probe only needs the final
top-1/top5 CIL number, which trainer.py prints unconditionally regardless of
those flags) -- no MetricsLogger/CE-ledger overhead.
"""
import json
import os

OUT_DIR = "exps/cllora_ease_tuna_rainbow_lr_refine"

METHOD_CFG = {
    "cllora": dict(
        model_name="cllora", backbone_type="vit_base_patch16_224_cllora",
        lr_fields=("init_lr", "later_lr"),
        epoch_fields=("init_epochs", "later_epochs"), epochs=20,
        batch_size=48, optimizer="sgd", scheduler="cosine", weight_decay=0.0001,
        min_lr=0, ffn_num=10, vpt_type="Deep", prompt_token_num=5,
        use_diagonal=False, recalc_sim=True, alpha=0.0, use_init_ptm=False,
        beta=0, use_old_data=False, use_reweight=True, moni_adam=False,
        adapter_num=-1, msa_adapt=True, use_distillation=True,
        use_block_weight=True, msa=[1, 0, 1], general_pos=[0, 1, 2, 3, 4, 5],
        specfic_pos=[6, 7, 8, 9, 10, 11],
    ),
    "tuna": dict(
        model_name="tuna", backbone_type="vit_base_patch16_224_tuna",
        lr_fields=("init_lr",),
        epoch_fields=("tuned_epoch",), epochs=20,
        batch_size=48, optimizer="adamw", scheduler="cosine", weight_decay=0.0005,
        min_lr=0, r=10, reinit_optimizer=True, init_milestones=[10],
        init_lr_decay=0.1, reg=0.001, use_orth=False, crct_epochs=30,
        ca_lr=0.005, ca_storage_efficient_method="covariance",
        ca_storage_efficient_method_choices=["covariance", "variance"],
        decay=False, drop=0.0, drop_path=0.0, scale=20.0, m=0.0,
    ),
    "ease": dict(
        model_name="ease", backbone_type="vit_base_patch16_224_ease",
        lr_fields=("init_lr", "later_lr"),
        epoch_fields=("init_epochs", "later_epochs"), epochs=20,
        batch_size=48, optimizer="adamw", scheduler="cosine", weight_decay=0.0005,
        min_lr=0, ffn_num=10, vpt_type="Deep", prompt_token_num=5,
        use_diagonal=False, recalc_sim=True, alpha=0.1, use_init_ptm=False,
        beta=0, use_old_data=False, use_reweight=True, moni_adam=False,
        adapter_num=-1,
    ),
    "rainbowprompt": dict(
        model_name="rainbowprompt", backbone_type="vit_base_patch16_224_rainbowprompt",
        lr_fields=("init_lr",),
        epoch_fields=("tuned_epoch",), epochs=20,
        batch_size=48, weight_decay=0.0, min_lr=0.0,
        rp_length=20, self_attn_idx=[0, 1, 2, 3, 4, 5], KI_iter=10,
        pull_constraint_coeff=0.1, use_linear=True, rp_D1=28, rp_D2=56,
    ),
}

# init_cls/increment match the wave1_final splits (same as the original sweep);
# stop_after_tasks is this sweep's OWN explicit user-specified truncation --
# NOT the same counts as the original cllora_ease_tuna_rainbow_sens grid.
DATASET_CFG = {
    "cifar224":        dict(init_cls=10, increment=10, stop_after_tasks=5),
    "sun397":          dict(init_cls=37, increment=40, stop_after_tasks=5),
    "food101":         dict(init_cls=6,  increment=5,  stop_after_tasks=8),
    "imagenetr":       dict(init_cls=10, increment=10, stop_after_tasks=8),
    "omnibenchmark1k": dict(init_cls=10, increment=10, stop_after_tasks=15),
}

ALL_DATASETS = ("cifar224", "imagenetr", "sun397", "food101", "omnibenchmark1k")
OTHER_DATASETS = ("cifar224", "imagenetr", "sun397", "food101")  # ALL_DATASETS minus omnibenchmark1k

# {method: {dataset: [lr, lr, ...]}} -- explicit values, user-specified 2026-08-11.
LR_GRID = {
    "rainbowprompt": {d: [1e-3, 5e-4, 1e-4] for d in ALL_DATASETS},
    "cllora": {
        **{d: [3e-2, 1e-2, 5e-2] for d in OTHER_DATASETS},
        "omnibenchmark1k": [3e-1, 5e-1, 7e-1],
    },
    "ease": {
        "cifar224": [5e-3, 1e-2, 3e-2],
        "food101": [5e-3, 1e-2, 3e-2],
        "sun397": [5e-3, 1e-2, 3e-2],
        "omnibenchmark1k": [3e-5, 1e-5],
    },
    "tuna": {
        "imagenetr": [5e-3, 8e-3, 1e-2],
        "omnibenchmark1k": [5e-3, 8e-3, 1e-2],
    },
}

SEED = 1993


def lr_tag(lr):
    """Filename-safe tag for an LR value, e.g. 0.0005 -> '5e-4', 0.3 -> '3e-1'."""
    return "{:.0e}".format(lr).replace("e-0", "e-").replace("e+0", "e")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for method, per_dataset_lrs in LR_GRID.items():
        mcfg = METHOD_CFG[method]
        for dataset, lrs in per_dataset_lrs.items():
            dcfg = DATASET_CFG[dataset]
            for lr in lrs:
                tag = lr_tag(lr)
                cfg = dict(
                    dataset=dataset,
                    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
                    init_cls=dcfg["init_cls"], increment=dcfg["increment"],
                    stop_after_tasks=dcfg["stop_after_tasks"],
                    seed=[SEED], scenario="cil",
                    pretrained=True, final_metrics=False, print_forget=False,
                    batch_size=mcfg["batch_size"], weight_decay=mcfg["weight_decay"],
                    min_lr=mcfg["min_lr"],
                    model_name=mcfg["model_name"], backbone_type=mcfg["backbone_type"],
                    device=["0"],
                    prefix="lr_refine_{}_{}_{}".format(method, dataset, tag),
                )
                for f in mcfg["epoch_fields"]:
                    cfg[f] = mcfg["epochs"]
                for f in mcfg["lr_fields"]:
                    cfg[f] = lr
                if "optimizer" in mcfg:
                    cfg["optimizer"] = mcfg["optimizer"]
                if "scheduler" in mcfg:
                    cfg["scheduler"] = mcfg["scheduler"]

                if method in ("cllora", "ease"):
                    cfg["vpt_type"] = mcfg["vpt_type"]
                    cfg["prompt_token_num"] = mcfg["prompt_token_num"]
                    cfg["ffn_num"] = mcfg["ffn_num"]
                    for k in ("use_diagonal", "recalc_sim", "alpha", "use_init_ptm",
                              "beta", "use_old_data", "use_reweight", "moni_adam",
                              "adapter_num"):
                        cfg[k] = mcfg[k]
                if method == "cllora":
                    for k in ("msa_adapt", "use_distillation", "use_block_weight",
                              "msa", "general_pos", "specfic_pos"):
                        cfg[k] = mcfg[k]
                if method == "tuna":
                    for k in ("r", "reinit_optimizer", "init_milestones",
                              "init_lr_decay", "reg", "use_orth", "crct_epochs",
                              "ca_lr", "ca_storage_efficient_method",
                              "ca_storage_efficient_method_choices", "decay",
                              "drop", "drop_path", "scale", "m"):
                        cfg[k] = mcfg[k]
                if method == "rainbowprompt":
                    for k in ("rp_length", "self_attn_idx", "KI_iter",
                              "pull_constraint_coeff", "use_linear", "rp_D1", "rp_D2"):
                        cfg[k] = mcfg[k]

                path = os.path.join(OUT_DIR, "{}_{}_{}.json".format(method, dataset, tag))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
