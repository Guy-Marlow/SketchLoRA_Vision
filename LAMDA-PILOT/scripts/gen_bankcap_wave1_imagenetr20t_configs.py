"""Config generator for the bankcap_wave1_imagenetr20t campaign (2026-08-14).

Two call modes, selected by which override flag is passed:

  --sketchlora-rank-cap N   -- 3 seed configs for SketchLoRA, base =
                               exps/wave1_final/sketchlora_imagenetr_s{seed}.json,
                               override sketchlora_rank_cap=N (all other wave1
                               settings -- svd_energy_target=0.01,
                               sketchlora_admission=bounded_eviction, etc. --
                               untouched).

  --bank-cap-mb X           -- 15 configs (5 methods x 3 seeds) for
                               O-LoRA/TreeLoRA/RainbowPrompt/CL-LoRA/EASE, in
                               that order, each with bank_cap_mb=X added.
                               O-LoRA/TreeLoRA/RainbowPrompt base off their
                               existing exps/wave1_final/*_imagenetr_s{seed}.json
                               configs. CL-LoRA/EASE have NO wave1_final
                               ImageNet-R config (they were never part of the
                               original wave1_final grid) -- built fresh here
                               instead, full 20-task/20-epoch/final_metrics=true,
                               mirroring wave1_final's other methods'
                               conventions, using the LRs settled by the
                               cllora_ease_tuna_lr_refine2 sweep (2026-08-14):
                               CL-LoRA locked at 0.05 for ImageNet-R; EASE
                               settled at 1e-2 uniformly. Flagged explicitly --
                               not pulled from any existing "official" config.

--phase-tag threads into `prefix` (e.g. "rank16", "rank16_capderived") so this
generator can be called twice per phase (sketchlora, then the derived-cap
others) without the two calls' run_logs/final output colliding.
"""

import argparse
import copy
import json
import os

WAVE1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exps", "wave1_final")
SEEDS = [1993, 1996, 1999]

# See module docstring -- no wave1_final ImageNet-R config exists for these
# two, so a full-length (20 task, 20 epoch, final_metrics=true) config is
# built here from scratch, mirroring wave1_final/olora_imagenetr_s*.json's
# shared conventions (batch_size 48, print_forget true) and the OTHER
# per-method fields straight from exps/cllora_ease_tuna_rainbow_sens's own
# "1x" configs (vpt_type/prompt_token_num/ffn_num/msa/general_pos/specfic_pos
# for CL-LoRA; the matching EASE fields) -- only dataset scope (stop_after_tasks
# removed -> full 20 tasks), final_metrics, and init_lr/later_lr are changed.
CLLORA_BASE = {
    "dataset": "imagenetr", "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
    "shuffle": True, "init_cls": 10, "increment": 10,
    "scenario": "cil", "pretrained": True, "print_forget": True, "final_metrics": True,
    "batch_size": 48, "weight_decay": 0.0001, "min_lr": 0,
    "model_name": "cllora", "backbone_type": "vit_base_patch16_224_cllora",
    "device": ["0"],
    "init_epochs": 20, "later_epochs": 20, "init_lr": 0.05, "later_lr": 0.05,
    "optimizer": "sgd", "scheduler": "cosine",
    "vpt_type": "Deep", "prompt_token_num": 5, "ffn_num": 10,
    "use_diagonal": False, "recalc_sim": True, "alpha": 0.0, "use_init_ptm": False,
    "beta": 0, "use_old_data": False, "use_reweight": True, "moni_adam": False,
    "adapter_num": -1, "msa_adapt": True, "use_distillation": True, "use_block_weight": True,
    "msa": [1, 0, 1], "general_pos": [0, 1, 2, 3, 4, 5], "specfic_pos": [6, 7, 8, 9, 10, 11],
}
EASE_BASE = {
    "dataset": "imagenetr", "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
    "shuffle": True, "init_cls": 10, "increment": 10,
    "scenario": "cil", "pretrained": True, "print_forget": True, "final_metrics": True,
    "batch_size": 48, "weight_decay": 0.0005, "min_lr": 0,
    "model_name": "ease", "backbone_type": "vit_base_patch16_224_ease",
    "device": ["0"],
    "init_epochs": 20, "later_epochs": 20, "init_lr": 0.01, "later_lr": 0.01,
    "optimizer": "adamw", "scheduler": "cosine",
    "vpt_type": "Deep", "prompt_token_num": 5, "ffn_num": 10,
    "use_diagonal": False, "recalc_sim": True, "alpha": 0.1, "use_init_ptm": False,
    "beta": 0, "use_old_data": False, "use_reweight": True, "moni_adam": False,
    "adapter_num": -1,
}

WAVE1_BASED_METHODS = {
    "olora": "olora_imagenetr_s{seed}.json",
    "treelora": "treelora_imagenetr_s{seed}.json",
    "rainbowprompt": "rainbowprompt_imagenetr_s{seed}.json",
}
FRESH_BASE_METHODS = {
    "cllora": CLLORA_BASE,
    "ease": EASE_BASE,
}
OTHER_METHODS_ORDER = ["olora", "treelora", "rainbowprompt", "cllora", "ease"]


def _load_base(method, seed):
    if method in WAVE1_BASED_METHODS:
        path = os.path.join(WAVE1, WAVE1_BASED_METHODS[method].format(seed=seed))
        with open(path) as f:
            return json.load(f)
    cfg = copy.deepcopy(FRESH_BASE_METHODS[method])
    cfg["seed"] = [seed]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--phase-tag", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--sketchlora-rank-cap", type=int)
    group.add_argument("--bank-cap-mb", type=float)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # phase_tag is folded into the FILENAME itself, not just `prefix` --
    # the SLURM driver's run_one() dedups by config basename alone (matching
    # every other campaign script in this project), so two phases (e.g.
    # rank16 vs rank32) producing same-named files in different directories
    # would silently collide on the SAME run_logs/*.out path and the second
    # phase's runs would be skipped as "already done". Filenames must be
    # globally unique across the whole campaign, not just within one phase's
    # output directory.
    if args.sketchlora_rank_cap is not None:
        for seed in SEEDS:
            path = os.path.join(WAVE1, "sketchlora_imagenetr_s{}.json".format(seed))
            with open(path) as f:
                cfg = json.load(f)
            cfg["sketchlora_rank_cap"] = args.sketchlora_rank_cap
            cfg["prefix"] = "bankcap_{}_s{}".format(args.phase_tag, seed)
            out = os.path.join(args.out_dir, "sketchlora_{}_s{}.json".format(args.phase_tag, seed))
            with open(out, "w") as f:
                json.dump(cfg, f, indent=2)
            print(out)
    else:
        for method in OTHER_METHODS_ORDER:
            for seed in SEEDS:
                cfg = _load_base(method, seed)
                cfg["bank_cap_mb"] = args.bank_cap_mb
                cfg["prefix"] = "bankcap_{}_s{}".format(args.phase_tag, seed)
                out = os.path.join(args.out_dir, "{}_{}_s{}.json".format(method, args.phase_tag, seed))
                with open(out, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(out)


if __name__ == "__main__":
    main()
