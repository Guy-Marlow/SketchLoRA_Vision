"""Plan B §B2 lr-sweep driver for Wave-1 methods (SeqLoRA/O-LoRA/InfLoRA/TreeLoRA).

REVISED 2026-07-23 for the 20-epoch confirmation re-sweep (impl_plan_7.23.2026,
Plan B §B2/§B1): epochs bumped 10->20 (adopted uniformly on all 5 datasets after
the SLURM CPU/RAM starvation fix voided the 10-epoch budget rationale); same lr
arm VALUES as the original 10-epoch sweep, for direct comparability. ImageNet-R's
validation split enlarged 10%->20% per §B2 (the prior ~120-img/task split had
+/-2-4pt binomial noise, below the decision resolution at this task count).
Flip-watch (pre-registered in the plan): InfLoRA/IN-R's 1.5e-3 win at 10 epochs
was justified as compensating a 5x epoch cut from its native 50; at 20 epochs
that's only a 2.5x cut, so the argument for needing a higher lr is weaker --
watch for it flipping back toward 5e-4 here.

Protocol (Plan B §B2): 3-point sweep {lr0/3, lr0, 3*lr0} per method, on CIFAR-100 +
ImageNet-R, 1 seed, selected on a validation split (stratified, carved BEFORE task
splitting, never touching test). CIFAR winner -> CIFAR/Food101/SUN397/Omni; IN-R
winner -> IN-R.

Reuses the real DataManager + Learner classes unchanged (same eval-routing path
already verified correct) -- the only addition is carving a validation split out of
the train pool in-place on the DataManager, via the existing
get_dataset_with_split() machinery, BEFORE the model ever sees the data. Test data
is never touched.

Usage: python scripts/run_lr_sweep_b2.py --method seqlora --dataset cifar224 --lr 0.001 --device 0
"""
import argparse
import json
import logging
import sys
import numpy as np
import torch

from utils.data_manager import DataManager
from utils import factory
from trainer import _set_device, _set_random

# Plan B §B3 structural fields (rank/alpha/method-specific knobs), lr filled by --lr
METHOD_CFG = {
    "seqlora": dict(model_name="seqlora", backbone_type="vit_base_patch16_224_lora",
                     lora_rank=10, lora_alpha=None, lora_merge=False),
    "olora": dict(model_name="olora", backbone_type="vit_base_patch16_224_lora",
                   lora_rank=10, lora_alpha=None, lora_merge=True,
                   lamda_1=0.5, lamda_2=0.0),
    "inflora": dict(model_name="inflora", backbone_type="vit_base_patch16_224_lora",
                      lora_rank=10, lora_alpha=None, lora_merge=True,
                      lamb=0.95, lame=1.0),
    "treelora": dict(model_name="treelora", backbone_type="vit_base_patch16_224_lora",
                       lora_rank=10, lora_alpha=None, reg=0.1),
    # Round 2 §3.1: SketchLoRA lr sweep, identical protocol to the others --
    # removes the previously-documented asymmetry (SketchLoRA's rate was
    # borrowed from SeqLoRA, never independently tuned). Frozen variant
    # throughout (cap 128, bounded eviction/conformant reading, LoRA wd 0,
    # eps=0.01), exact SVD per §2.5 (Plan A §A5.3 never cleared).
    "sketchlora": dict(model_name="sketchlora", backbone_type="vit_base_patch16_224_lora",
                        lora_rank=10, lora_alpha=None, lora_merge=True, lora_train_merge=True,
                        svd_rank=10, svd_oversampling=10, svd_energy_target=0.01,
                        lora_n_slots=2, merge_op="exactsvd",
                        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
                        sketchlora_lora_wd=0.0),
}

DATASET_CFG = {
    "cifar224": dict(init_cls=10, increment=10, epochs=20),
    "imagenetr": dict(init_cls=10, increment=10, epochs=20),
}

SMOKE_TASKS = 3
SEED = 1993
VAL_FRAC = {"cifar224": 0.10, "imagenetr": 0.20}  # IN-R enlarged per Plan B §B2


def carve_validation_split(dm, val_frac):
    """Stratified val split from the train pool, in-place, BEFORE any task split.
    Test data is never read/touched here."""
    counts = [np.sum(dm._train_targets == c) for c in range(dm.nb_classes)]
    val_per_class = max(1, round(val_frac * min(counts)))
    train_ds, val_ds = dm.get_dataset_with_split(
        list(range(dm.nb_classes)), source="train", mode="test",
        val_samples_per_class=val_per_class,
    )
    dm._train_data, dm._train_targets = train_ds.images, train_ds.labels
    # redirect "test" source to the carved validation set -- eval_task()/_eval_cnn
    # read self.test_loader built from source="test", so this is the ONLY line
    # that makes selection happen against validation instead of real test.
    dm._test_data, dm._test_targets = val_ds.images, val_ds.labels
    logging.info("[lr_sweep] carved val split: {} val/class, {} val total, {} train remaining".format(
        val_per_class, len(val_ds.labels), len(train_ds.labels)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHOD_CFG.keys()))
    p.add_argument("--dataset", required=True, choices=list(DATASET_CFG.keys()))
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--device", required=True)
    args_cli = p.parse_args()

    mcfg = METHOD_CFG[args_cli.method]
    dcfg = DATASET_CFG[args_cli.dataset]

    args = dict(
        dataset=args_cli.dataset,
        memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
        init_cls=dcfg["init_cls"], increment=dcfg["increment"],
        seed=SEED, scenario="cil", stop_after_tasks=SMOKE_TASKS,
        pretrained=True, print_forget=True, final_metrics=True,
        tuned_epoch=dcfg["epochs"], batch_size=48, init_lr=args_cli.lr,
        weight_decay=0.0005, min_lr=0.0,
        device=[args_cli.device],
        prefix="b2_sweep_20ep_{}_{}_lr{}".format(args_cli.method, args_cli.dataset, args_cli.lr),
    )
    args.update(mcfg)
    args["seed"] = SEED  # _set_random / model code expect scalar after trainer.py's per-seed unwrap

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [b2_sweep] %(message)s",
                         handlers=[logging.StreamHandler(sys.stdout)])

    _set_random(args["seed"])
    _set_device(args)  # converts args["device"] to torch.device list, matches trainer.py

    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"],
                                args["init_cls"], args["increment"], args)
    carve_validation_split(data_manager, VAL_FRAC[args_cli.dataset])

    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    n_run = min(args["stop_after_tasks"], data_manager.nb_tasks)
    val_curve = []
    for task in range(n_run):
        model.incremental_train(data_manager)
        cnn_accy, _ = model.eval_task()
        model.after_task()
        val_curve.append(cnn_accy["top1"])
        logging.info("task {} val_top1={:.2f}".format(task, cnn_accy["top1"]))

    result = dict(method=args_cli.method, dataset=args_cli.dataset, lr=args_cli.lr,
                   val_curve=val_curve, val_avg=sum(val_curve) / len(val_curve))
    out = "run_logs/b2_sweep_20ep/{}_{}_lr{}.json".format(args_cli.method, args_cli.dataset, args_cli.lr)
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    logging.info("[b2_sweep] DONE val_avg={:.2f} -> {}".format(result["val_avg"], out))


if __name__ == "__main__":
    main()
