"""Generate task-incremental (pure task-boundary) CIL smoke configs for the full 11-method
roster, on ONE ordering of the first 5 tasks of the imagenet-r 20-task split (init_cls=10,
increment=10, seed=1993, shuffle=true -> one fixed shuffled class order).

Purpose: identify which methods collapse under real (not truncated) per-method epoch counts,
so review can focus on the ones that need it. Each method's own already-validated HPs (from
exps/final/*_cifar224_20t.json and exps/matched_*.json) are reused verbatim except for
dataset/init_cls/increment/stop_after_tasks/prefix/device -- no HP retuning here.
"""
import json
import os

OUT_DIR = "exps/review/task_incremental_imr5t"
COMMON = {
    "dataset": "imagenetr",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": True,
    "init_cls": 10,
    "increment": 10,
    "seed": [1993],
    "scenario": "cil",
    "stop_after_tasks": 5,
    "pretrained": True,
    "print_forget": True,
    "final_metrics": True,
}

METHODS = {
    "seqlora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "seqlora", "lora_merge": False,
    },
    "olora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "olora", "lora_merge": True, "lamda_1": 0.5, "lamda_2": 0.0,
    },
    "inflora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "inflora", "lora_merge": True, "lamb": 0.95, "lame": 1.0,
    },
    "sketchlora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "sketchlora", "lora_merge": True, "lora_train_merge": True,
        "svd_rank": 8, "svd_oversampling": 10, "svd_energy_target": 0.01, "lora_n_slots": 2,
    },
    "hidelora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "hidelora", "lora_momentum": 0.1, "reg": 0.001,
        "crct_epochs": 30, "ca_lr": 0.005, "n_centroids": 10,
    },
    "treelora": {
        "backbone_type": "vit_base_patch16_224_lora",
        "lora_rank": 8, "lora_alpha": 32, "tuned_epoch": 10, "init_lr": 0.0003,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0.0,
        "model_name": "treelora", "reg": 0.5,
    },
    "rainbowprompt": {
        "backbone_type": "vit_base_patch16_224_rainbowprompt",
        "tuned_epoch": 20, "init_lr": 0.03, "batch_size": 48,
        "weight_decay": 0.0, "min_lr": 0.0,
        "model_name": "rainbowprompt", "rp_length": 20,
        "self_attn_idx": [0, 1, 2, 3, 4, 5], "KI_iter": 10,
        "pull_constraint_coeff": 0.1, "use_linear": True, "rp_D1": 28, "rp_D2": 56,
    },
    "progprompt": {
        "backbone_type": "vit_base_patch16_224_progprompt",
        "tuned_epoch": 5, "init_lr": 0.0015, "batch_size": 48,
        "weight_decay": 0.0, "min_lr": 0.0,
        "model_name": "progprompt", "prompt_len": 10,
    },
    "ease": {
        "backbone_type": "vit_base_patch16_224_ease",
        "init_epochs": 20, "init_lr": 0.025, "later_epochs": 20, "later_lr": 0.025,
        "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0,
        "optimizer": "sgd", "scheduler": "cosine",
        "model_name": "ease", "vpt_type": "Deep", "prompt_token_num": 5, "ffn_num": 64,
        "use_diagonal": False, "recalc_sim": True, "alpha": 0.1, "use_init_ptm": False,
        "beta": 0, "use_old_data": False, "use_reweight": True,
        "moni_adam": False, "adapter_num": -1,
    },
    "cllora": {
        "backbone_type": "vit_base_patch16_224_in21k_cllora",
        "init_epochs": 30, "init_lr": 0.03, "later_epochs": 30, "later_lr": 0.03,
        "batch_size": 64, "weight_decay": 0.0001, "min_lr": 0,
        "optimizer": "sgd", "scheduler": "cosine",
        "model_name": "cllora", "vpt_type": "Deep", "prompt_token_num": 5, "ffn_num": 8,
        "use_diagonal": False, "recalc_sim": True, "alpha": 0.0, "use_init_ptm": False,
        "beta": 0, "use_old_data": False, "use_reweight": True, "moni_adam": False,
        "adapter_num": -1, "msa_adapt": True, "use_distillation": True,
        "use_block_weight": True, "msa": [1, 0, 1],
        "general_pos": [0, 1, 2, 3, 4, 5], "specfic_pos": [6, 7, 8, 9, 10, 11],
    },
    "tuna": {
        "backbone_type": "vit_base_patch16_224_in21k_tuna",
        "tuned_epoch": 15, "init_lr": 0.01, "batch_size": 16, "weight_decay": 0.0005,
        "min_lr": 0, "optimizer": "sgd", "scheduler": "cosine",
        "reinit_optimizer": True, "init_milestones": [10], "init_lr_decay": 0.1,
        "model_name": "tuna", "reg": 0.001, "use_orth": False,
        "crct_epochs": 30, "ca_lr": 0.005,
        "ca_storage_efficient_method": "covariance",
        "ca_storage_efficient_method_choices": ["covariance", "variance"],
        "decay": False, "drop": 0.0, "drop_path": 0.0, "r": 16, "scale": 20.0, "m": 0.0,
    },
}

# 2 confirmed-safe free GPUs on this box right now (0/2 in use by other users' jobs;
# 3 is the tiny 4GB DGX display card; 4 needs CUDA_DEVICE_ORDER=PCI_BUS_ID to map correctly).
DEVICE_CYCLE = ["1", "4"]

os.makedirs(OUT_DIR, exist_ok=True)
for i, (name, hp) in enumerate(METHODS.items()):
    cfg = dict(COMMON)
    cfg.update(hp)
    cfg["device"] = [DEVICE_CYCLE[i % len(DEVICE_CYCLE)]]
    cfg["prefix"] = "review_imr5t_{}".format(name)
    path = os.path.join(OUT_DIR, "{}.json".format(name))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", path)
