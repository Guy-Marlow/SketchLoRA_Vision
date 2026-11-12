"""CL-LoRA-Sketch-FoldFirst + LoRA-Subtraction + LoRA-Subtraction-Sketch
campaign (2026-11-12 user request): three sequential sub-campaigns, run in
this order -- (1) CL-LoRA-Sketch-FoldFirst, (2) LoRA-Subtraction (growing
bank, log_bank_rank=true), (3) LoRA-Subtraction-Sketch (fixed-size sketch,
no bank-rank logging -- not applicable to a fixed-size structure).

5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) x 3 methods = 45 runs.

Every config's HYPERPARAMETERS are built by importing and calling the EXACT
settled builder function from the campaign that already established that
method's settings -- byte-identical values, not re-derived. `prefix` is
handled differently per method, though:

  - cllora_sketch_foldfirst -- scripts/gen_sketch_foldfirst_wave1_configs.py
    ::cllora_sketch_foldfirst_cfg (the "v3" design: orth_a_side=true,
    use_block_weight_scaling=false, sketchlora_align_weight=0.75), prefix
    LEFT UNCHANGED from that source campaign (same "split/reorganized for
    submission reasons" convention scripts/gen_sketch_foldfirst_wave1_
    sdlora_cllora_configs.py already uses) -- safe here because every prior
    attempt under that tag crashed with KeyError('use_distillation') before
    this session's fix (scripts/gen_sketch_foldfirst_wave1_configs.py's
    CLLORA_COMMON was missing use_distillation/alpha/beta, both now added
    and verified), so no metrics file was ever written under it and there
    is no stale "done" data it could collide with.
  - lorasub -- scripts/gen_bank_rank_wave1_configs.py::lorasub_cfg
    (init_lr: cifar224=1e-3, everything else=3e-3; weight_decay=0.0;
    svd_thres=0.99, use_proto_repulsion=true, proto_repulsion_weight=0.2,
    proto_margin=2.0; log_bank_rank=true, bank_rank_energy_target=0.99),
    prefix OVERRIDDEN to a fresh cllora_lorasub_wave1-specific tag (see
    _lorasub_cfg_fresh below) -- NOT safe to reuse bank_rank_wave1's own
    prefix: all 15 of ITS runs already completed (status=="done", dated
    2026-11-11) BEFORE this session's DRS-probe/init/prototype fix to
    models/lorasub.py's _update_protos/_proto_repulsion_loss (commit
    8800724 -- changes how prototype vectors are computed, a term added to
    the training loss every batch from task 1 onward whenever
    use_proto_repulsion=true). Reusing that prefix would let scripts/
    _bankcap_run_done.py's resumability check skip these as already-done
    and silently report stale, pre-fix numbers as if they validated the
    fixed code -- the exact bug found and fixed 2026-11-12.
  - lorasub_sketch -- scripts/gen_lorasub_sdlora_sketch_wave1_configs.py
    ::lorasub_sketch_cfg (same per-dataset init_lr as plain lorasub above,
    lora_n_slots=2, svd_rank=10, merge_op="qrsvd"; no log_bank_rank -- a
    fixed 2-slot sketch has no growing bank to measure), prefix OVERRIDDEN
    the same way and for the same reason -- lorasub_sdlora_sketch_wave1's
    own 15 runs are equally stale relative to models/lorasub_sketch.py's
    own fixes in the same commit.

Per-dataset splits (init_cls/increment), identical across all three
methods and to every other wave1-style campaign in this project:
cifar224=10/10 (10 tasks), imagenetr=10/10 (20 tasks),
omnibenchmark1k=10/10 (100 tasks), sun397=37/40 (10 tasks),
food101=6/5 (20 tasks) -- enforced by construction here since all three
imported builder functions share the identical DATASETS dict shape from
their own source modules.
"""
import json
import os

from gen_sketch_foldfirst_wave1_configs import cllora_sketch_foldfirst_cfg
from gen_bank_rank_wave1_configs import lorasub_cfg
from gen_lorasub_sdlora_sketch_wave1_configs import lorasub_sketch_cfg

OUT_BASE = "exps/cllora_lorasub_wave1"
SEEDS = [1993, 1996, 1999]
DATASETS = ["cifar224", "food101", "imagenetr", "omnibenchmark1k", "sun397"]


# lorasub/lorasub_sketch get FRESH prefixes here, unlike
# cllora_sketch_foldfirst_cfg above -- the already-"done" bank_rank_wave1_
# lorasub_*/lorasub_sdlora_sketch_wave1_lorasub_sketch_* metrics (dated
# 2026-11-11) were produced BEFORE this session's DRS-probe/init/prototype
# fix to models/lorasub.py's _update_protos/_proto_repulsion_loss (commit
# 8800724 -- changes how prototype vectors are computed, a term added to
# the training loss every batch from task 1 onward whenever
# use_proto_repulsion=true, which every settled config here requires).
# Reusing the OLD prefix would let scripts/_bankcap_run_done.py's
# resumability check skip these as already-done and silently report
# stale, pre-fix numbers as if they validated the fixed code -- exactly
# the bug found 2026-11-12. cllora_sketch_foldfirst_cfg's prefix is left
# unchanged from its own source campaign: every prior attempt under that
# tag crashed before writing a metrics file, so there is no stale "done"
# data it could collide with.
def _lorasub_cfg_fresh(ds_name, seed):
    cfg = lorasub_cfg(ds_name, seed)
    cfg["prefix"] = "cllora_lorasub_wave1_lorasub_{}_s{}".format(ds_name, seed)
    return cfg


def _lorasub_sketch_cfg_fresh(ds_name, seed):
    cfg = lorasub_sketch_cfg(ds_name, seed)
    cfg["prefix"] = "cllora_lorasub_wave1_lorasub_sketch_{}_s{}".format(ds_name, seed)
    return cfg


BUILDERS = {
    "cllora_sketch_foldfirst": cllora_sketch_foldfirst_cfg,
    "lorasub": _lorasub_cfg_fresh,
    "lorasub_sketch": _lorasub_sketch_cfg_fresh,
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
