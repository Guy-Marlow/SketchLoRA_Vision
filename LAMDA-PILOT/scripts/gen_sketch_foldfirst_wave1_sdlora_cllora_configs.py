"""SD-LoRA-Sketch-FoldFirst + CL-LoRA-Sketch-FoldFirst re-run (2026-11-11
user request) -- a SPLIT-OUT subset of the original scripts/gen_sketch_
foldfirst_wave1_configs.py campaign, submitted separately because
LoRA-Subtraction-Sketch-FoldFirst is NOT ready yet (its underlying code --
backbone/vit_lorasub.py, models/lorasub.py, models/lorasub_sketch.py -- has
today's DRS-probe/init/prototype fixes applied locally but UNCOMMITTED and
UNPUSHED, still being smoke-tested). SD-LoRA-Sketch-FoldFirst and
CL-LoRA-Sketch-FoldFirst have no such pending fix -- both were already
smoke-tested clean and pushed (commit ed10e84) -- so there is no reason to
hold them back.

Hyperparameters are imported directly from the original generator (NOT
re-derived) to guarantee byte-identical configs to what that campaign would
have produced for these two methods -- this is genuinely the same logical
campaign, just split across two SLURM submissions for scheduling reasons,
so `prefix` (and therefore the final run_logs/final/<model>/metrics_*.json
naming) is deliberately left unchanged from the original generator's own
convention.

LoRA-Subtraction-Sketch-FoldFirst will be re-run later, together with
plain (growing-bank) LoRA-Subtraction, under its own separate SLURM file
once the fixes above are committed, pushed, and validated.
"""
import json
import os

from gen_sketch_foldfirst_wave1_configs import (
    DATASETS, SEEDS, sdlora_sketch_foldfirst_cfg, cllora_sketch_foldfirst_cfg,
)

OUT_BASE = "exps/sketch_foldfirst_wave1_sdlora_cllora"

BUILDERS = {
    "sdlora_sketch_foldfirst": sdlora_sketch_foldfirst_cfg,
    "cllora_sketch_foldfirst": cllora_sketch_foldfirst_cfg,
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
