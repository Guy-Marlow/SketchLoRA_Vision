"""Completes the wave1_final grid (2026-08-19 user request): production-scale
(full task counts, 3 seeds, final_metrics=true) configs for CL-LoRA/EASE/TUNA
at their settled learning rates -- no prior campaign ever ran these three at
full length; every existing exps/cllora_ease_tuna_* directory is a truncated
LR sweep/refinement (see other_methods_config_and_lr_sweep memory for the
settlement history) -- plus a fresh O-LoRA rerun on the wave1_final grid to
get current-code persistent-memory numbers (O-LoRA's persistent_state() was
reworked in commit 23c6405, 2026-08-10, removing the dense-fold accounting;
wave1_final's ORIGINAL O-LoRA runs, 2026-08-05, predate that fix -- same
rationale as the already-completed wave1_rerun_treelora campaign, which
covered TreeLoRA but explicitly scoped O-LoRA out at the time per user
request).

Settled learning rates (other_methods_config_and_lr_sweep memory,
2026-08-13 decision, confirmed final):
  CL-LoRA: 0.05 uniformly across all 5 benchmarks.
  EASE:    1e-2 uniformly across all 5 benchmarks.
  TUNA:    5e-3 on ImageNet-R and OmniBenchmark-1K, 1e-3 elsewhere
           (CIFAR-100, Food-101, SUN397).

Method/architecture hyperparameters (optimizer, scheduler, batch size,
epochs, rank/ffn-dim, etc.) copied VERBATIM from
scripts/gen_cllora_ease_tuna_lr_refine2_configs.py's own METHOD_CFG -- only
lr, dataset split (full task counts, not truncated), seed count (3, not 1),
and final_metrics/print_forget (on, not off -- this is a production run, not
a quick sweep) change here.

Dataset splits match wave1_final exactly (see wave1_final_campaign memory):
CIFAR-100 10/10 (10 tasks), Food-101 6/5 (20 tasks), ImageNet-R 10/10
(20 tasks), SUN397 37/40 (10 tasks), OmniBenchmark-1K 10/10 (100 tasks).

O-LoRA rerun configs are copied verbatim from exps/wave1_final/olora_*.json
(same pattern as scripts/gen_wave1_rerun_treelora_configs.py) -- only prefix
changes, to avoid the stale-metrics-skip trap a reused prefix would cause on
a resumed submission.
"""
import copy
import json
import os

OUT_DIR = "exps/wave1_final_completion"
WAVE1_FINAL = "exps/wave1_final"
SEEDS = [1993, 1996, 1999]

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

# Full (untruncated) wave1_final splits -- no stop_after_tasks field.
DATASET_CFG = {
    "cifar224":        dict(init_cls=10, increment=10),
    "sun397":          dict(init_cls=37, increment=40),
    "imagenetr":       dict(init_cls=10, increment=10),
    "food101":         dict(init_cls=6,  increment=5),
    "omnibenchmark1k": dict(init_cls=10, increment=10),
}
# cheapest-first (real measured O-LoRA per-dataset wall time this project,
# see bankcap_wave1_imagenetr20t_campaign memory): sun397 < imagenetr <
# cifar224 < food101 << omnibenchmark1k.
DATASET_ORDER = ["sun397", "imagenetr", "cifar224", "food101", "omnibenchmark1k"]

# {method: {dataset: lr}} -- settled values, other_methods_config_and_lr_sweep memory.
SETTLED_LR = {
    "cllora": {d: 0.05 for d in DATASET_CFG},
    "ease": {d: 0.01 for d in DATASET_CFG},
    "tuna": {
        "imagenetr": 0.005, "omnibenchmark1k": 0.005,
        "cifar224": 0.001, "food101": 0.001, "sun397": 0.001,
    },
}


def gen_cllora_ease_tuna(out_dir):
    written = []
    for method, mcfg in METHOD_CFG.items():
        for dataset in DATASET_ORDER:
            dcfg = DATASET_CFG[dataset]
            lr = SETTLED_LR[method][dataset]
            for seed in SEEDS:
                cfg = dict(
                    dataset=dataset,
                    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
                    init_cls=dcfg["init_cls"], increment=dcfg["increment"],
                    seed=[seed], scenario="cil",
                    pretrained=True, final_metrics=True, print_forget=True,
                    batch_size=mcfg["batch_size"], weight_decay=mcfg["weight_decay"],
                    min_lr=mcfg["min_lr"],
                    model_name=mcfg["model_name"], backbone_type=mcfg["backbone_type"],
                    device=["0"],
                    prefix="wave1_final_completion_{}_{}_s{}".format(method, dataset, seed),
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

                path = os.path.join(out_dir, "{}_{}_s{}.json".format(method, dataset, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    return written


def gen_olora_rerun(out_dir):
    written = []
    for dataset in DATASET_ORDER:
        for seed in SEEDS:
            src = os.path.join(WAVE1_FINAL, "olora_{}_s{}.json".format(dataset, seed))
            cfg = copy.deepcopy(json.load(open(src)))
            assert cfg.get("final_metrics") is True, \
                "{} does not have final_metrics=true -- refusing to silently rerun without it".format(src)
            cfg["prefix"] = "wave1_final_completion_olora_{}_s{}".format(dataset, seed)
            out = os.path.join(out_dir, "olora_{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(out, "w"), indent=2)
            written.append(out)
    return written


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = gen_olora_rerun(OUT_DIR) + gen_cllora_ease_tuna(OUT_DIR)
    print("wrote {} configs to {}/ (olora rerun: 15, cllora/ease/tuna: {} each x 5 datasets x 3 seeds)".format(
        len(written), OUT_DIR, len(SEEDS)))


if __name__ == "__main__":
    main()
