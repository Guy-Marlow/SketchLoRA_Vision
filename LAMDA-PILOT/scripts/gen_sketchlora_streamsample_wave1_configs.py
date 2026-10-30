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

3 chunk sizes (500/1000/1500 images) x 2 datasets x 3 seeds = 18 runs.
Chunk sizes chosen per explicit user instruction (2026-10-29), at a scale
large enough to produce a real experiment (dozens of chunks per run, not
the smoke test's throwaway 200-sample/21-chunk scale): imagenetr's ~26-27k
full-regime images give ~52/27/18 chunks at 500/1000/1500; cifar224's 50k
give ~100/50/33.

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
CHUNK_SIZES = [500, 1000, 1500]

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


def main():
    written = []
    for dataset, src_path in SOURCE_CONFIGS.items():
        out_dir = os.path.join(OUT_DIR, dataset)
        os.makedirs(out_dir, exist_ok=True)
        base = json.load(open(src_path))
        for key in STRIP_KEYS:
            base.pop(key, None)

        for chunk in CHUNK_SIZES:
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
