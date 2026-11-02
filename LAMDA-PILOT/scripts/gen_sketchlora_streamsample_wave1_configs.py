"""Boundary-agnostic sample-count streaming campaign, wave 1 (2026-10-29 user
request): canonical SketchLoRA (sketchlora_align, merge_op="qrsvd",
sketchlora_align_mode="orth", sketchlora_align_weight=0.5 -- the unqualified
"SketchLoRA" convention since 2026-10-28, see models/sketchlora_align.py's
own module docstring) evaluated under models/stream_mixin.py::stream_run's
REPAIRED unique-image budget design (boundary_mode="sample") -- real task
boundaries exist in the data but training proceeds on fixed-size disjoint
chunks of stream_chunk_samples images, irrespective of where those
boundaries fall. NOT the faulty epoch-repeated-sample-count clock
(boundary_mode="sample_legacy") -- see trainer.py lines 99-113 for the
dispatch, confirmed clean/separate before this campaign was built.

Verified via a 3-task/200-sample-chunk smoke test (exps/smoke/
sketchlora_align_sample_stream_test.json, 2026-10-29) that:
  1. stream_chunk_samples dispatches correctly (direct image count, not the
     legacy stream_budget_mb-derived conversion).
  2. models/sketchlora_align.py's _stream_extra_loss override (added this
     session -- previously MISSING, meaning every prior streaming-mode
     sketchlora_align run silently trained with NO orthogonality penalty at
     all) is actually wired in: models/stream_mixin.py::_stream_train_epoch
     unconditionally adds its return value to the loss whenever it isn't a
     literal float 0.0, and _reg_terms()'s own gate (self._cur_task <= 0)
     opens from the 2nd chunk onward -- confirmed directly from the smoke
     test's own SketchDiag log lines, whose "task N" counter climbs once
     PER CHUNK (0->20 over 21 chunks), not once per real task.

Base configs are this session's top-1% sketchlora_align LR-sweep configs
(scripts/sketchlora_top1_lr_sweep.py) -- same backbone/lora_rank/svd_rank/
batch_size/weight_decay as the master-table wave1 configs, just generated
by a different script -- EXCEPT init_lr, which is overridden back to
exps/sketchlora_orth_wave1_datasets/'s own 1e-3 for both datasets
(2026-10-30 user instruction: don't introduce an LR difference as a second
variable on top of the streaming-vs-oracle comparison this campaign is
actually testing; the top-1%-tuned winners, 2e-3/imagenetr and 9e-4/
cifar224, were a separate, unrelated tuning effort and not reused here).
Both stripped of stop_after_tasks (that LR search truncated to 10/5 tasks
for speed; this campaign runs the FULL regime -- 20 tasks/imagenetr,
10 tasks/cifar224, matching exps/sketchlora_orth_wave1_datasets/'s own
master-table convention of no stop_after_tasks key at all) and of
final_metrics (inert under boundary_mode="sample" anyway -- trainer.py
returns from _run_stream before the final_metrics/MetricsLogger block is
ever reached, confirmed by direct code reading, not assumption).

Originally 3 chunk sizes (500/1000/1500 images) x 2 datasets (imagenetr/
cifar224) x 3 seeds = 18 runs. EXTENDED 2026-11-02 (explicit user request,
after analyzing the first 18 runs -- see Documents/
sketchlora_streamsample_wave1_cifar100_accuracy.png): imagenetr is left
UNCHANGED at its original 3 chunk sizes (not part of this request); cifar224
gains chunk 2000/2500; omnibenchmark1k is added as a THIRD dataset at all 5
chunk sizes (500/1000/1500/2000/2500) -- never run under this campaign
before. Confirmed with the user: don't re-run cifar224's existing 500/1000/
1500 (already complete and analyzed) just to reach the new grid uniformly --
regenerating them here is harmless (byte-identical to what's already on
disk, same source/formula/LR, so the SLURM script's own resumability check
still skips them) rather than something this generator needs to special-case.

omnibenchmark1k has NO existing qrsvd+orth streaming-convention source
config to copy from (exps/wave1_final/sketchlora_omnibenchmark1k_s1993.json
is base "sketchlora" with ADAPTIVE admission -- a different variant, not
canonical SketchLoRA) -- built inline instead (OMNIBENCHMARK1K_BASE),
matching the same field set as the imagenetr/cifar224 source configs
exactly, same init_lr=0.001 (confirmed matching exps/wave1_final/
sketchlora_omnibenchmark1k_s1993.json's own init_lr, so this isn't a new LR
decision). No stop_after_tasks -- full regime, same "no cap" convention as
imagenetr/cifar224 -- NOTE: omnibenchmark1k is a 100-task dataset (~10x
cifar224's task count); its full-regime image count could not be confirmed
locally (data not downloaded on this machine) before building these
configs -- walltime in the .slurm script is a rough estimate, not measured.

sketchlora_diag_dir is set to a campaign-dedicated directory -- cross-
campaign sketch_diag collisions are a known, previously-fixed bug class in
this project (see memory: sketch_diag collision bug fixed for wave1_final
and sketchlora_sens_ablations_wave2 separately); every new campaign gets
its own dir rather than relying on the shared default.
"""
import json
import os

OUT_DIR = "exps/sketchlora_streamsample_wave1"
SEEDS = [1993, 1996, 1999]

# per-dataset chunk-size grid -- imagenetr untouched, cifar224/omnibenchmark1k
# extended per the 2026-11-02 request.
DATASET_CHUNK_SIZES = {
    "imagenetr": [500, 1000, 1500],
    "cifar224": [500, 1000, 1500, 2000, 2500],
    "omnibenchmark1k": [500, 1000, 1500, 2000, 2500],
}

SOURCE_CONFIGS = {
    "imagenetr": "exps/sketchlora_top1_lr_sweep/imagenetr/2e-3_s1993.json",
    "cifar224": "exps/sketchlora_top1_lr_sweep/cifar224/9e-4_s1993.json",
}

# 2026-10-30: use the SAME init_lr as the master-table wave1 configs
# (exps/sketchlora_orth_wave1_datasets/{imagenetr,cifar224}_s1993.json, both
# 1e-3) rather than each dataset's own top-1%-tuned winner -- avoids
# introducing a second variable alongside the streaming-vs-oracle comparison.
WAVE1_INIT_LR = 0.001

# keys from the LR-sweep source configs that don't belong in a full-regime
# streaming config -- stripped unconditionally before adding the streaming
# keys below.
STRIP_KEYS = ("stop_after_tasks", "final_metrics", "sketchlora_diag_dir", "prefix")

# 2026-11-02: no qrsvd+orth streaming-convention source config exists for
# omnibenchmark1k (see module docstring) -- built directly to match the
# exact same field set as the imagenetr/cifar224 source configs after their
# own STRIP_KEYS pass.
OMNIBENCHMARK1K_BASE = {
    "dataset": "omnibenchmark1k",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": True,
    "init_cls": 10,
    "increment": 10,
    "scenario": "cil",
    "pretrained": True,
    "print_forget": True,
    "tuned_epoch": 20,
    "batch_size": 48,
    "weight_decay": 0.0005,
    "min_lr": 0.0,
    "model_name": "sketchlora_align",
    "backbone_type": "vit_base_patch16_224_lora",
    "lora_rank": 10,
    "lora_alpha": None,
    "lora_merge": True,
    "lora_train_merge": True,
    "svd_rank": 10,
    "svd_oversampling": 10,
    "lora_n_slots": 2,
    "sketch_diag": True,
    "sketchlora_lora_wd": 0.0,
    "sketchlora_align_mode": "orth",
    "sketchlora_align_weight": 0.5,
    "merge_op": "qrsvd",
    "device": ["0"],
}


def load_base(dataset):
    if dataset in SOURCE_CONFIGS:
        base = json.load(open(SOURCE_CONFIGS[dataset]))
        for key in STRIP_KEYS:
            base.pop(key, None)
        return base
    return dict(OMNIBENCHMARK1K_BASE)


def main():
    written = []
    for dataset, chunk_sizes in DATASET_CHUNK_SIZES.items():
        out_dir = os.path.join(OUT_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        base = load_base(dataset)

        for chunk in chunk_sizes:
            for seed in SEEDS:
                cfg = dict(base)
                cfg["seed"] = [seed]
                cfg["init_lr"] = WAVE1_INIT_LR
                cfg["boundary_mode"] = "sample"
                cfg["stream_chunk_samples"] = chunk
                cfg["final_metrics"] = False
                cfg["sketchlora_diag_dir"] = "run_logs/sketchlora_streamsample_wave1/{}/diag".format(dataset)
                cfg["prefix"] = "sketchlora_streamsample_wave1_{}_chunk{}_s{}".format(dataset, chunk, seed)
                path = os.path.join(out_dir, "chunk{}_s{}.json".format(chunk, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)

    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
