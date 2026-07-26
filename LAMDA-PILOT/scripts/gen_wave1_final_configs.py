"""Finalized Wave-1 configs (SeqLoRA/O-LoRA/InfLoRA/TreeLoRA) per impl_plan_7.23.2026's
Plan A/B, LOCKED after tonight's Plan B §B2 20-epoch confirmation re-sweep (validation-
split selected, 3-task smoke, CIFAR-100 + ImageNet-R, seed 1993, 20 epochs/task):

User-locked decision (2026-07-23, post-resweep):
  ImageNet-R: 1e-3 for ALL FOUR methods -- a deliberate simplification over the
    sweep's own per-method result (InfLoRA's hi=1.5e-3 decisively won there, and
    O-LoRA's center/hi were an exact tie at 1e-3/3e-3) in favor of one shared value.
  CIFAR-100: method-specific, exactly matching each method's own 20-epoch sweep
    winner -- SeqLoRA 3e-4 (lo won), O-LoRA 1e-3 (center won), InfLoRA 5e-4
    (center won), TreeLoRA 1e-3 (center won).
  Food101/SUN397/Omni-1K: no dedicated sweep exists for these three (Plan B §B2's
    protocol only covers CIFAR-100+ImageNet-R) -- per the plan's own propagation
    rule ("CIFAR winner -> CIFAR/Food101/SUN397/Omni"), they inherit each method's
    CIFAR-100 value above. FLAG: this supersedes an earlier, separately-established
    Food101-specific override (5e-4 for SeqLoRA/O-LoRA/TreeLoRA, from a genuine
    standalone Food101 HP search predating tonight's sweep) for methods whose
    locked CIFAR-100 value differs from that earlier 5e-4 finding (O-LoRA and
    TreeLoRA, both now 1e-3 on CIFAR-100 -- SeqLoRA's is unaffected, since 3e-4 and
    the old Food101 override happen to differ too, 3e-4 vs 5e-4). Revisit if a
    dedicated Food101/SUN397/Omni sweep is run later.

Epochs: 20/task on ALL FIVE datasets (Plan B §B1, adopted after the SLURM CPU/RAM
throughput fix voided the original 10-epoch budget rationale -- supersedes the
earlier 10-epoch-short/15-epoch-Omni split).

Canonical splits (Plan A §A1): CIFAR-100 B0Inc10(10T); ImageNet-R B0Inc10(20T);
Food101 B6Inc5(20T); SUN397 B37Inc40(10T); OmniBenchmark-1K B0Inc10(100T).
Batch 48 uniform.
"""
import json
import os

OUT_DIR = "exps/wave1_final"

# lr_cifar propagates to food101/sun397/omnibenchmark1k per Plan B's stated rule;
# lr_imagenetr is the user-locked uniform 1e-3, method-specific field kept only
# for clarity (all four are the same value).
METHOD_CFG = {
    "seqlora": dict(model_name="seqlora", backbone_type="vit_base_patch16_224_lora",
                     lora_rank=10, lora_alpha=None, lora_merge=False,
                     lr_cifar=0.0003, lr_imagenetr=0.001),
    "olora": dict(model_name="olora", backbone_type="vit_base_patch16_224_lora",
                   lora_rank=10, lora_alpha=None, lora_merge=True,
                   lamda_1=0.5, lamda_2=0.0,
                   lr_cifar=0.001, lr_imagenetr=0.001),
    "inflora": dict(model_name="inflora", backbone_type="vit_base_patch16_224_lora",
                      lora_rank=10, lora_alpha=None, lora_merge=True,
                      lamb=0.95, lame=1.0,
                      lr_cifar=0.0005, lr_imagenetr=0.001),
    "treelora": dict(model_name="treelora", backbone_type="vit_base_patch16_224_lora",
                       lora_rank=10, lora_alpha=None, reg=0.1,
                       lr_cifar=0.001, lr_imagenetr=0.001),
}

DATASET_CFG = {
    "cifar224": dict(init_cls=10, increment=10, epochs=20),
    "imagenetr": dict(init_cls=10, increment=10, epochs=20),
    "food101": dict(init_cls=6, increment=5, epochs=20),
    "sun397": dict(init_cls=37, increment=40, epochs=20),
    "omnibenchmark1k": dict(init_cls=10, increment=10, epochs=20),
}

SEEDS = [1993, 1996, 1999]  # Plan A §A1 class-order seeds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for method, mcfg in METHOD_CFG.items():
        for dataset, dcfg in DATASET_CFG.items():
            lr = mcfg["lr_imagenetr"] if dataset == "imagenetr" else mcfg["lr_cifar"]
            for seed in SEEDS:
                cfg = dict(
                    dataset=dataset,
                    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
                    init_cls=dcfg["init_cls"], increment=dcfg["increment"],
                    seed=[seed], scenario="cil",
                    pretrained=True, print_forget=True, final_metrics=True,
                    tuned_epoch=dcfg["epochs"], batch_size=48, init_lr=lr,
                    weight_decay=0.0005, min_lr=0.0,
                    model_name=mcfg["model_name"], backbone_type=mcfg["backbone_type"],
                    lora_rank=mcfg["lora_rank"], lora_alpha=mcfg["lora_alpha"],
                    device=["0"],
                    prefix="wave1_final_{}_{}_s{}".format(method, dataset, seed),
                )
                if "lora_merge" in mcfg:
                    cfg["lora_merge"] = mcfg["lora_merge"]
                if method == "olora":
                    cfg["lamda_1"] = mcfg["lamda_1"]
                    cfg["lamda_2"] = mcfg["lamda_2"]
                if method == "inflora":
                    cfg["lamb"] = mcfg["lamb"]
                    cfg["lame"] = mcfg["lame"]
                if method == "treelora":
                    cfg["reg"] = mcfg["reg"]

                path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(method, dataset, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
