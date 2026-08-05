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

EXTENDED 2026-08-05 (user request): added SketchLoRA. Unlike the four methods
above, SketchLoRA does NOT get a CIFAR-propagates-to-Food/SUN/Omni lr rule --
per explicit user instruction, it uses ONE uniform lr (0.001) across all five
datasets, taken directly from its own already-completed 100-task OmniBenchmark-1k
run (exps/round2_slurm_grid/sketchlora_100mb_s1993.json, init_lr=0.001 --
despite that file's leftover "omni30t" prefix from an earlier naming convention,
it is actually stop_after_tasks=100/init_cls=10/increment=10, i.e. the 100-task
oracle-CIL split, not a 30-task bounded-memory run) and its ImageNet-R oracle
config (exps/sketchlora_ablations_imagenetr20t/sketchlora_current_s*.json),
which already independently used the same 0.001 -- confirmed consistent, not
an arbitrary transplant. All other SketchLoRA hyperparameters (rank=10,
merge_op=randsvd, svd_energy_target=0.01, sketchlora_admission=bounded_eviction,
rank_cap=128) are unchanged from that current/live ImageNet-R config -- the user
did not ask to retune those, only to extend the lr choice to the other 4
datasets. cifar224/food101/sun397/imagenetr configs written by a prior manual
pass on 2026-08-05 are superseded by (byte-for-byte reproduced via) this
generator on the same day, for provenance consistency with the other 4 methods.
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
    "sketchlora": dict(model_name="sketchlora", backbone_type="vit_base_patch16_224_lora",
                         lora_rank=10, lora_alpha=None, lora_merge=True,
                         lora_train_merge=True, svd_rank=10, svd_oversampling=10,
                         lora_n_slots=2, sketch_diag=True, merge_op="randsvd",
                         svd_energy_target=0.01, sketchlora_admission="bounded_eviction",
                         sketchlora_rank_cap=128, sketchlora_lora_wd=0.0,
                         sketchlora_eviction_reading="conformant",
                         # uniform across all 5 datasets -- see module docstring
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
                if method == "sketchlora":
                    # sketchlora_lora_wd=0.0 (below) already zeroes weight decay on the
                    # LoRA/sketch param group specifically (models/sketchlora.py::
                    # _optimizer_param_groups splits LoRA params into their own
                    # zero-decay group once sketchlora_lora_wd is set) -- that's the
                    # ONLY thing that needed zeroing (2026-08-05 user request: "we just
                    # don't want the sketch decaying"). The classifier head correctly
                    # keeps the generic top-level weight_decay=0.0005 (same as every
                    # other method) via that same override's "other_params" group --
                    # a 2026-08-05 attempt to also zero the head's decay was REVERTED
                    # same day per direct user correction: head decay was already
                    # intentional, not a gap to close.
                    for k in ("lora_train_merge", "svd_rank", "svd_oversampling",
                              "lora_n_slots", "sketch_diag", "merge_op",
                              "svd_energy_target", "sketchlora_admission",
                              "sketchlora_rank_cap", "sketchlora_lora_wd",
                              "sketchlora_eviction_reading"):
                        cfg[k] = mcfg[k]

                path = os.path.join(OUT_DIR, "{}_{}_s{}.json".format(method, dataset, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
