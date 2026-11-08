"""Bank-rank phase 2 campaign (2026-11-09 user request): three sequential
sub-campaigns, run in this order --

  1. sdlora       -- vanilla SD-LoRA, 5 datasets x 3 seeds, log_bank_rank=true
                     (models/sdlora.py's own summed-bank effective-rank
                     diagnostic, added this session -- see that file's
                     _log_bank_effective_rank docstring for why its delta
                     construction differs from O-LoRA/LoRA-Subtraction's).
  2. cllora       -- vanilla CL-LoRA, 5 datasets x 3 seeds, log_bank_rank=true
                     (models/cllora.py's own diagnostic, also added this
                     session -- a HYPOTHETICAL "what if the per-task
                     specific_pos adapters were folded into one dense
                     update" capacity proxy, since CL-LoRA's real forward
                     never sums them; see that file's docstring).
  3. cllora_sketch -- the CL-LoRA-Sketch "v3" design (orth_a_side=true,
                     use_block_weight_scaling=false) promoted to a full-
                     length, full-dataset campaign, with
                     sketchlora_align_weight RAISED to 0.75 (was 0.5 in the
                     original smoke-scale ablation, exps/cllora_sketch_wave1/
                     v3_orth_only_s1993.json) per explicit 2026-11-09 user
                     instruction. No log_bank_rank here -- the whole point of
                     this phase is accuracy/memory at full length, not the
                     rank diagnostic (which only matters for the GROWING-bank
                     comparison in phases 1-2), same reasoning as models/
                     lorasub_sketch.py's own no-op override.

Hyperparameters for sdlora/cllora are copied byte-for-byte (minus
log_bank_rank/bank_rank_energy_target) from this project's own already-
settled per-dataset conventions:
  - sdlora: exps/slao_sdlora_wave1_datasets/sdlora/<dataset>_s1993.json
    (init_lr per dataset: cifar224/food101/omnibenchmark1k/sun397=5e-3,
    imagenetr=8e-3; tuned_epoch=20, batch_size=48, weight_decay=5e-4,
    lora_rank=10).
  - cllora: exps/wave1_final_completion/cllora_<dataset>_s1993.json
    (init_lr=later_lr=0.05 uniformly, init_epochs=later_epochs=20,
    batch_size=48, weight_decay=1e-4, sgd/cosine, ffn_num=10,
    msa=[1,0,1], general_pos=[0..5], specfic_pos=[6..11] -- identical
    across all 5 datasets, only dataset/init_cls/increment vary).
  - cllora_sketch: same architecture constants as cllora (ffn_num=10,
    msa/general_pos/specfic_pos), plus svd_rank=10, merge_op="qrsvd",
    orth_a_side=true, use_block_weight_scaling=false,
    sketchlora_align_weight=0.75 (raised from the smoke-scale 0.5),
    init_lr=later_lr=0.05 (same as vanilla cllora -- no separate tune
    exists yet for the sketch variant itself), weight_decay=1e-4,
    20 epochs/task, full task count (no stop_after_tasks, unlike the
    5-task ablation smoke runs this config is promoted from).

Per-dataset splits (init_cls/increment), identical across all three
sub-campaigns and to every other wave1-style campaign in this project:
cifar224=10/10 (10 tasks), imagenetr=10/10 (20 tasks),
omnibenchmark1k=10/10 (100 tasks), sun397=37/40 (10 tasks),
food101=6/5 (20 tasks).
"""
import json
import os

OUT_BASE = "exps/bankrank_phase2_wave1"
SEEDS = [1993, 1996, 1999]

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
    "omnibenchmark1k": dict(dataset="omnibenchmark1k", init_cls=10, increment=10),
    "sun397": dict(dataset="sun397", init_cls=37, increment=40),
    "food101": dict(dataset="food101", init_cls=6, increment=5),
}

SDLORA_LR = {
    "cifar224": 0.005, "imagenetr": 0.008, "omnibenchmark1k": 0.005,
    "sun397": 0.005, "food101": 0.005,
}

CLLORA_COMMON = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
    batch_size=48, weight_decay=0.0001, min_lr=0,
    init_epochs=20, later_epochs=20, init_lr=0.05, later_lr=0.05,
    optimizer="sgd", scheduler="cosine", vpt_type="Deep", prompt_token_num=5,
    ffn_num=10, use_diagonal=False, recalc_sim=True, alpha=0.0,
    use_init_ptm=False, beta=0, use_old_data=False, use_reweight=True,
    moni_adam=False, adapter_num=-1, msa_adapt=True, use_distillation=True,
    msa=[1, 0, 1], general_pos=[0, 1, 2, 3, 4, 5], specfic_pos=[6, 7, 8, 9, 10, 11],
    device=["0"],
)


def sdlora_cfg(ds_name, seed):
    cfg = dict(
        memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
        scenario="cil", pretrained=True, print_forget=True, final_metrics=True,
        tuned_epoch=20, batch_size=48, weight_decay=0.0005, min_lr=0.0,
        lora_rank=10, lora_alpha=None, device=["0"],
        model_name="sdlora", backbone_type="vit_base_patch16_224_sdlora",
        lora_merge=True, lora_train_merge=True,
        log_bank_rank=True, bank_rank_energy_target=0.99,
    )
    cfg.update(DATASETS[ds_name])
    cfg.update(seed=[seed], init_lr=SDLORA_LR[ds_name],
               prefix="bankrank_phase2_wave1_sdlora_{}_s{}".format(ds_name, seed))
    return cfg


def cllora_cfg(ds_name, seed):
    cfg = dict(CLLORA_COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], model_name="cllora", backbone_type="vit_base_patch16_224_cllora",
        use_block_weight=True,
        log_bank_rank=True, bank_rank_energy_target=0.99,
        prefix="bankrank_phase2_wave1_cllora_{}_s{}".format(ds_name, seed),
    )
    return cfg


def cllora_sketch_cfg(ds_name, seed):
    cfg = dict(CLLORA_COMMON)
    cfg.update(DATASETS[ds_name])
    cfg.update(
        seed=[seed], model_name="cllora_sketch",
        backbone_type="vit_base_patch16_224_cllora_sketch",
        svd_rank=10, merge_op="qrsvd",
        use_block_weight_scaling=False, orth_a_side=True,
        sketchlora_align_weight=0.75,
        prefix="bankrank_phase2_wave1_cllora_sketch_{}_s{}".format(ds_name, seed),
    )
    # use_block_weight/use_init_ptm/recalc_sim/moni_adam/adapter_num/
    # use_diagonal/use_old_data/use_reweight are all dead keys for this
    # model (see models/cllora_sketch.py -- it never reads them), kept out
    # entirely rather than carried over from CLLORA_COMMON as no-ops.
    for dead_key in ("use_diagonal", "recalc_sim", "use_init_ptm", "use_old_data",
                      "use_reweight", "moni_adam", "adapter_num", "vpt_type",
                      "prompt_token_num", "msa_adapt"):
        cfg.pop(dead_key, None)
    return cfg


BUILDERS = {
    "sdlora": sdlora_cfg,
    "cllora": cllora_cfg,
    "cllora_sketch": cllora_sketch_cfg,
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
