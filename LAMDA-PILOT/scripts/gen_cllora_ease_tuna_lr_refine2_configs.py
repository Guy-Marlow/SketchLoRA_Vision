"""Third-round LR sensitivity sweep for CL-LoRA/EASE/TUNA (follow-up to
scripts/gen_lr_sweep_configs.py's original multiplier sweep and
scripts/gen_cllora_ease_tuna_rainbow_lr_refine_configs.py's explicit-value
refinement). User-specified 2026-08-13, explicit rationale per method:

  CL-LoRA: ImageNet-R and Food-101 are DONE -- lr=0.05 is locked in as final
    for both, no further testing, not part of this sweep's LR_GRID at all.
    What's left: (a) OmniBenchmark-1K never had a usable result (the previous
    round's 0.3/0.5/0.7 all collapsed near-random at 15 tasks) -- probe
    0.01/0.03/0.05 instead, an order of magnitude below the collapsed range
    and in line with what worked on the other 4 datasets; (b) a collapse
    check for lr=0.05 (the value now locked for ImageNet-R/Food-101) on
    CIFAR-100 and SUN397, to catch the same kind of instability CL-LoRA/Omni
    showed before committing to long runs elsewhere.

  EASE: the previous round's OmniBenchmark-1K winner (1e-5) is 3 orders of
    magnitude below Food-101's winner (1e-2) -- suspicious, since
    OmniBenchmark-1K's smaller per-task sample count (~1690/task vs Food-101's
    ~3750-4500/task) should if anything favor a LARGER effective LR per step,
    not a vastly smaller one; the previous round's OmniBenchmark-1K probe
    also only tested 2 points (1e-5, 3e-5), both far below every other
    dataset's tested range, so the earlier "winner" is thin evidence. Test
    0.001/0.005/0.01 -- EASE's normal range -- on ALL FIVE datasets, to get a
    genuinely comparable read.

  TUNA: only ever tested on ImageNet-R and OmniBenchmark-1K across both prior
    rounds (never CIFAR-100/Food-101/SUN397). Test 0.001/0.005/0.01 on all
    five to fill that gap and get full coverage before locking in.

TASK COUNTS (2026-08-13, the new standard going forward -- longer than
either prior round's, chosen to be "solid" for downstream long-run
decisions): CIFAR-100=5, SUN397=5, Food-101=10, ImageNet-R=10,
OmniBenchmark-1K=20 tasks. init_cls/increment unchanged from every other
campaign in this project.

Method/architecture hyperparameters (optimizer, scheduler, batch size,
epochs, rank/ffn-dim, etc.) copied VERBATIM from
scripts/gen_cllora_ease_tuna_rainbow_lr_refine_configs.py's METHOD_CFG --
nothing about the training loop changes here, only LR values, datasets, and
task-count truncation. RainbowPrompt dropped entirely (not part of this
round's ask).

Single seed (1993), same as both prior rounds. final_metrics/print_forget
both off -- a short LR probe only needs the final top-1/top5 CIL number
(trainer.py prints it unconditionally), no MetricsLogger/CE-ledger overhead
needed.
"""
import json
import os

OUT_DIR = "exps/cllora_ease_tuna_lr_refine2"

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
}

DATASET_CFG = {
    "cifar224":        dict(init_cls=10, increment=10, stop_after_tasks=5),
    "sun397":          dict(init_cls=37, increment=40, stop_after_tasks=5),
    "food101":         dict(init_cls=6,  increment=5,  stop_after_tasks=10),
    "imagenetr":       dict(init_cls=10, increment=10, stop_after_tasks=10),
    "omnibenchmark1k": dict(init_cls=10, increment=10, stop_after_tasks=20),
}

ALL_DATASETS = ("cifar224", "imagenetr", "sun397", "food101", "omnibenchmark1k")

# {method: {dataset: [lr, lr, ...]}} -- explicit values, user-specified 2026-08-13.
LR_GRID = {
    "cllora": {
        "omnibenchmark1k": [0.01, 0.03, 0.05],
        "cifar224": [0.05],
        "sun397": [0.05],
        # imagenetr/food101 intentionally absent -- locked at 0.05, no rerun.
    },
    "ease": {d: [0.001, 0.005, 0.01] for d in ALL_DATASETS},
    "tuna": {d: [0.001, 0.005, 0.01] for d in ALL_DATASETS},
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
                    prefix="lr_refine2_{}_{}_{}".format(method, dataset, tag),
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

                path = os.path.join(OUT_DIR, "{}_{}_{}.json".format(method, dataset, tag))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
