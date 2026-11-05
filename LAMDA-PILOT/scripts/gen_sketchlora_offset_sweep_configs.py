"""Epoch-offset consolidation sweep for vanilla SketchLoRA (2026-11-06 user
request): model_name=sketchlora_align, merge_op="qrsvd", sketchlora_align_
mode="orth" (this project's current canonical "SketchLoRA" default, see
memory sketchlora-orth-is-now-default-convention) -- identical config to
the sketchlora_qrsvd_fillout campaign's own main5 cells (same base
hyperparameters, same diagnostics), except boundary_mode="offset_
consolidation" instead of the plain oracle per-task loop.

offset_k is held at 20 (== tuned_epoch, "a fixed epoch count for each
segment" per the user's own framing when this driver was designed --
NOT swept) -- only offset_phi is swept, across {1,2,3,4,5}. phi=0 (the
"aligned" case: fold exactly at every task boundary, equivalent to this
same method's ordinary oracle run) is NOT included -- this sweep is
specifically about the phase-OFFSET cases.

2 datasets (CIFAR-100-10t, ImageNet-R-20t) x 5 offsets x 3 seeds
(1993/1996/1999) = 30 runs. Full-length runs (no stop_after_tasks) -- a
real campaign, not a truncated tuning probe.

Every diagnostic this project's other SketchLoRA campaigns carry is kept:
print_forget (now genuinely supported under offset_consolidation_run --
see 2026-11-06 fix to models/stream_mixin.py::_stream_eval/
offset_consolidation_run, which previously had NO forgetting computation
at all under any boundary_mode streaming driver), sketch_diag=True +
sketchlora_diag_dir (per-campaign directory, so this sweep's own fold-by-
fold retained-energy/fro diagnostics can't collide with any other
campaign's sketch_diag files), final_metrics=False (matching this
project's own convention for every OTHER boundary_mode streaming config --
final_metrics'/CE-profiling machinery is oracle-loop-only, never wired
into any streaming driver, so True here would be a silent no-op at best).
"""
import json
import os

OUT_BASE = "exps/sketchlora_offset_sweep"
RUN_LOGS_BASE = "run_logs/sketchlora_offset_sweep"
SEEDS = [1993, 1996, 1999]
OFFSETS = [1, 2, 3, 4, 5]
EPOCHS = 20   # tuned_epoch == offset_k, "default epoch count" (fixed per segment)

DATASETS = {
    "cifar224": dict(dataset="cifar224", init_cls=10, increment=10),
    "imagenetr": dict(dataset="imagenetr", init_cls=10, increment=10),
}

BASE = dict(
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    scenario="cil", pretrained=True, print_forget=True, final_metrics=False,
    tuned_epoch=EPOCHS, batch_size=48, init_lr=0.001, weight_decay=0.0005, min_lr=0.0,
    model_name="sketchlora_align", backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None, lora_merge=True, lora_train_merge=True,
    svd_rank=10, svd_oversampling=10, lora_n_slots=2, sketch_diag=True,
    sketchlora_lora_wd=0.0, sketchlora_align_mode="orth", sketchlora_align_weight=0.5,
    merge_op="qrsvd", device=["0"],
    boundary_mode="offset_consolidation", offset_k=EPOCHS,
)


def main():
    n = 0
    for ds_name, ds_kwargs in DATASETS.items():
        out_dir = os.path.join(OUT_BASE, ds_name)
        # Forward slashes explicitly (not os.path.join) -- this path is a
        # VALUE embedded in the JSON, read back by Python running on the
        # Linux SLURM cluster, not just a local file-write path on this
        # Windows generator machine. os.path.join here would silently bake
        # in backslashes that are meaningless path separators on Linux.
        diag_dir = "{}/{}/diag".format(RUN_LOGS_BASE, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        for phi in OFFSETS:
            for seed in SEEDS:
                cfg = dict(BASE)
                cfg.update(ds_kwargs)
                cfg["seed"] = [seed]
                cfg["offset_phi"] = phi
                prefix = "sketchlora_offset_sweep_{}_phi{}_s{}".format(ds_name, phi, seed)
                cfg["prefix"] = prefix
                cfg["sketchlora_diag_dir"] = diag_dir
                out_path = os.path.join(out_dir, "phi{}_s{}.json".format(phi, seed))
                json.dump(cfg, open(out_path, "w"), indent=2)
                n += 1
    print("wrote {} configs under {}".format(n, OUT_BASE))


if __name__ == "__main__":
    main()
