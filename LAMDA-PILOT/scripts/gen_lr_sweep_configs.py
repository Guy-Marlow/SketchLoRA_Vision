"""LR sweep for CL-LoRA/TUNA/EASE/RainbowPrompt (2026-08-05 user request): these
four methods have never had a tuned lr on the wave1_final 5-dataset convention --
this generates the 80-cell grid (4 methods x 5 datasets x 4 lr multipliers, single
seed, truncated task counts) used to pick one.

Architecture/training-loop settings are the "native config" already agreed and
applied to each method's exps/final/<method>_cifar224_20t.json base (2026-08-05
session): 20 epochs, batch 48, each method's own native optimizer, rank/ffn-dim
10 where that concept is real (TUNA's `r`, CL-LoRA/EASE's `ffn_num` -- NOT TUNA's
ffn_num, which is a hardcoded-16 dead field, see utils/inc_net.py's `_tuna`
branch), RainbowPrompt's rp_length=20 (already correct, untouched). CL-LoRA/EASE's
prompt_token_num/vpt_type fields are left exactly as in the base config and NOT
swept or otherwise touched -- utils/inc_net.py hardcodes vpt_on=False for both,
so those fields are inert regardless of value (user-confirmed 2026-08-05: "if
it's off by default, leave it off").

Dataset splits match the wave1_final convention exactly (same init_cls/increment
as exps/wave1_final/, NOT the older cifar224_20t 5/5 split these methods'
base configs still carry) -- CIFAR-100-10t/ImageNet-R-20t/Food101-20t/SUN397-10t/
OmniBenchmark-1k-100t. stop_after_tasks truncates each to a short prefix of that
split for sweep speed, per explicit user counts: cifar224 3, imagenetr 5, sun397
3, food101 4, omnibenchmark1k 10.

final_metrics/print_forget are both OFF (user: "we don't need any logging
enabled... we simply need the hyperparameters and the final top-5 CIL") --
trainer.py prints the CNN top1/top5 curve and "Average Accuracy (CNN)"
unconditionally regardless of those flags, so the number this sweep needs is
still recoverable straight from the .out log with no MetricsLogger/CE-ledger
overhead.
"""
import json
import os

OUT_DIR = "exps/cllora_ease_tuna_rainbow_sens"

METHOD_CFG = {
    "cllora": dict(
        model_name="cllora", backbone_type="vit_base_patch16_224_cllora",
        base_lr=0.03, lr_fields=("init_lr", "later_lr"),
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
        base_lr=0.0005, lr_fields=("init_lr",),
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
        base_lr=0.0005, lr_fields=("init_lr", "later_lr"),
        epoch_fields=("init_epochs", "later_epochs"), epochs=20,
        batch_size=48, optimizer="adamw", scheduler="cosine", weight_decay=0.0005,
        min_lr=0, ffn_num=10, vpt_type="Deep", prompt_token_num=5,
        use_diagonal=False, recalc_sim=True, alpha=0.1, use_init_ptm=False,
        beta=0, use_old_data=False, use_reweight=True, moni_adam=False,
        adapter_num=-1,
    ),
    "rainbowprompt": dict(
        model_name="rainbowprompt", backbone_type="vit_base_patch16_224_rainbowprompt",
        base_lr=0.03, lr_fields=("init_lr",),
        epoch_fields=("tuned_epoch",), epochs=20,
        batch_size=48, weight_decay=0.0, min_lr=0.0,
        rp_length=20, self_attn_idx=[0, 1, 2, 3, 4, 5], KI_iter=10,
        pull_constraint_coeff=0.1, use_linear=True, rp_D1=28, rp_D2=56,
    ),
}

# matches exps/wave1_final/'s splits exactly -- NOT the older cifar224_20t 5/5
# split these methods' base configs still carry.
DATASET_CFG = {
    "cifar224":        dict(init_cls=10, increment=10, stop_after_tasks=3),
    "imagenetr":       dict(init_cls=10, increment=10, stop_after_tasks=5),
    "sun397":          dict(init_cls=37, increment=40, stop_after_tasks=3),
    "food101":         dict(init_cls=6,  increment=5,  stop_after_tasks=4),
    "omnibenchmark1k": dict(init_cls=10, increment=10, stop_after_tasks=10),
}

# (tag, multiplier) -- order matches the user's own listing.
LR_MULTIPLIERS = [("1x", 1), ("2x", 2), ("10x", 10), ("0.1x", 0.1)]

SEED = 1993


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for method, mcfg in METHOD_CFG.items():
        for dataset, dcfg in DATASET_CFG.items():
            for lr_tag, mult in LR_MULTIPLIERS:
                lr = mcfg["base_lr"] * mult
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
                    prefix="lr_sweep_{}_{}_{}".format(method, dataset, lr_tag),
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

                path = os.path.join(OUT_DIR, "{}_{}_{}.json".format(method, dataset, lr_tag))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
