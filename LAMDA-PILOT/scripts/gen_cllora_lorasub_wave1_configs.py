"""CL-LoRA-Sketch-FoldFirst + LoRA-Subtraction + LoRA-Subtraction-Sketch
campaign (2026-11-12 user request): three sequential sub-campaigns, run in
this order -- (1) CL-LoRA-Sketch-FoldFirst, (2) LoRA-Subtraction (growing
bank, log_bank_rank=true), (3) LoRA-Subtraction-Sketch (fixed-size sketch,
no bank-rank logging -- not applicable to a fixed-size structure).

5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) x 3 methods = 45 runs.

Every config is built by importing and calling the EXACT settled builder
function from the campaign that already established that method's
hyperparameters -- byte-identical configs, not re-derived values, same
"split/reorganized for submission reasons" convention already used by
scripts/gen_sketch_foldfirst_wave1_sdlora_cllora_configs.py. `prefix` is
therefore deliberately LEFT UNCHANGED from each source campaign's own
convention (not re-tagged with this campaign's name) -- this is genuinely
the same logical runs, just re-submitted together in a specific order, so
the run_logs/final/<model>/metrics_<tag>.json identity (and therefore
scripts/_bankcap_run_done.py's resumability check) stays tied to whichever
of these three sub-campaigns a run logically belongs to:

  - cllora_sketch_foldfirst -- scripts/gen_sketch_foldfirst_wave1_configs.py
    ::cllora_sketch_foldfirst_cfg (the "v3" design: orth_a_side=true,
    use_block_weight_scaling=false, sketchlora_align_weight=0.75). This is
    the variant that crashed with KeyError('use_distillation') before this
    session's fix (scripts/gen_sketch_foldfirst_wave1_configs.py's
    CLLORA_COMMON was missing use_distillation/alpha/beta, both now added
    and verified) -- importing the now-fixed function here guarantees this
    campaign uses the corrected configs, not a second copy that could drift.
  - lorasub -- scripts/gen_bank_rank_wave1_configs.py::lorasub_cfg
    (init_lr: cifar224=1e-3, everything else=3e-3; weight_decay=0.0;
    svd_thres=0.99, use_proto_repulsion=true, proto_repulsion_weight=0.2,
    proto_margin=2.0; log_bank_rank=true, bank_rank_energy_target=0.99).
    Depends on backbone/vit_lorasub.py + models/lorasub.py's DRS-probe/
    init/prototype fixes, smoke-validated and committed this session
    (commit 8800724).
  - lorasub_sketch -- scripts/gen_lorasub_sdlora_sketch_wave1_configs.py
    ::lorasub_sketch_cfg (same per-dataset init_lr as plain lorasub above,
    lora_n_slots=2, svd_rank=10, merge_op="qrsvd"; no log_bank_rank -- a
    fixed 2-slot sketch has no growing bank to measure). Depends on
    models/lorasub_sketch.py's own fixes, same commit.

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

BUILDERS = {
    "cllora_sketch_foldfirst": cllora_sketch_foldfirst_cfg,
    "lorasub": lorasub_cfg,
    "lorasub_sketch": lorasub_sketch_cfg,
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
