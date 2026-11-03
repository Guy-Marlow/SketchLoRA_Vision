"""SLAO and SD-LoRA, full oracle (real task-boundary) regime, across all
five master-table datasets (2026-11-02 user request) -- a SEPARATE
experiment from sketchlora_streamsample_wave1: this measures vanilla SLAO/
SD-LoRA's own standing accuracy on each dataset (to gauge them against
SketchLoRA's own master-table numbers), with NO streaming/boundary_mode
involved at all. Neither method is run under boundary_mode="sample".

Settled learning rates (2026-10-28/2026-10-30, see memory:
slao_sdlora_settled_lr_convention.md and scripts/lr_tune_slao_then_sdlora*.py's
own driver logs):
  - imagenetr: own tuned winner -- SLAO=8e-4, SD-LoRA=8e-3
  - cifar224:  own tuned winner -- SLAO=1e-3, SD-LoRA=5e-3
  - sun397, food101, omnibenchmark1k: NEVER tuned on this machine (SUN397/
    Food-101 were never attempted at all; OmniBenchmark-1k crashed
    immediately, no local data) -- per explicit user decision, use
    cifar224's winning LR for both methods rather than a fresh tune. This
    matches the pattern already observed across this project's OTHER
    methods (exps/wave1_final/, exps/wave1_final_completion/): when a
    method's CIFAR-100 and ImageNet-R settled LRs differ (as they do here
    for both SLAO and SD-LoRA), SUN397/Food-101/OmniBenchmark-1k matched
    CIFAR-100's value in every case checked except one (TUNA's
    OmniBenchmark-1k, which matched ImageNet-R's instead) -- CIFAR-100 was
    the more common match, hence this choice.

Base configs are each method's own WINNING config from the LR-tuning
campaigns (exps/{slao,sdlora}_lr_tune_{cifar224,imagenetr}/), stripped of
stop_after_tasks (that search truncated to 5/10 tasks for speed; this
campaign runs the FULL regime) -- final_metrics/print_forget left ON
(unlike the LR search itself, which also had both on; this is a production
run, matching exps/wave1_final/'s own final_metrics=true convention).

BUG FIX (2026-11-04, user-caught): this previously hardcoded init_cls=10/
increment=10 for EVERY dataset, with a docstring claim that this matched
"exps/wave1_final/'s own 5-dataset grid" -- that claim was never actually
checked against the real files and was false. wave1_final (and the master-
table main5 campaign, which agrees with it) uses PER-DATASET splits, not a
uniform 10/10: confirmed directly from exps/wave1_final/sketchlora_{ds}_
s1993.json for all five --
  imagenetr: 10/10 (20 tasks)   cifar224: 10/10 (10 tasks)
  sun397:    37/40 (10 tasks)   food101:  6/5 (20 tasks)
  omnibenchmark1k: 10/10 (20 tasks)
Only cifar224/imagenetr/omnibenchmark1k happen to genuinely be 10/10, so the
old hardcoded value was silently wrong for sun397 and food101 specifically
-- those two datasets' already-run SLAO/SD-LoRA results (food101: all 3
seeds; sun397: 1 seed) used a DIFFERENT task split than SketchLoRA's own
convention and are not directly comparable task-index for task-index.
Those need to be regenerated with the fix below and re-run.

Per-dataset init_cls/increment now read directly from the matching
exps/wave1_final/sketchlora_{dataset}_s1993.json file at generation time
(not hardcoded a second time here) so this can't drift out of sync again.

5 datasets x 2 methods x 3 seeds = 30 runs.
"""
import json
import os

OUT_DIR = "exps/slao_sdlora_wave1_datasets"
SEEDS = [1993, 1996, 1999]
DATASETS = ["imagenetr", "cifar224", "sun397", "food101", "omnibenchmark1k"]
WAVE1_FINAL_REF = "exps/wave1_final/sketchlora_{dataset}_s1993.json"

SOURCE_CONFIGS = {
    "slao": "exps/slao_lr_tune_cifar224/slao_lr_tune_cifar224_1e-3_s1993.json",
    "sdlora": "exps/sdlora_lr_tune_cifar224/sdlora_lr_tune_cifar224_5e-3_s1993.json",
}

# own tuned winner for imagenetr/cifar224; CIFAR-100's own winner (0.001 for
# slao, 0.005 for sdlora) reused for the three untuned datasets.
LR_BY_DATASET = {
    "slao": {
        "imagenetr": 0.0008,
        "cifar224": 0.001,
        "sun397": 0.001,
        "food101": 0.001,
        "omnibenchmark1k": 0.001,
    },
    "sdlora": {
        "imagenetr": 0.008,
        "cifar224": 0.005,
        "sun397": 0.005,
        "food101": 0.005,
        "omnibenchmark1k": 0.005,
    },
}

STRIP_KEYS = ("stop_after_tasks", "prefix")


def main():
    written = []
    for method, src_path in SOURCE_CONFIGS.items():
        base = json.load(open(src_path))
        for key in STRIP_KEYS:
            base.pop(key, None)
        out_dir = os.path.join(OUT_DIR, method)
        os.makedirs(out_dir, exist_ok=True)

        for dataset in DATASETS:
            ref = json.load(open(WAVE1_FINAL_REF.format(dataset=dataset)))
            for seed in SEEDS:
                cfg = dict(base)
                cfg["dataset"] = dataset
                cfg["seed"] = [seed]
                cfg["init_cls"] = ref["init_cls"]
                cfg["increment"] = ref["increment"]
                cfg["init_lr"] = LR_BY_DATASET[method][dataset]
                cfg["prefix"] = "slao_sdlora_wave1_datasets_{}_{}_s{}".format(method, dataset, seed)
                path = os.path.join(out_dir, "{}_s{}.json".format(dataset, seed))
                json.dump(cfg, open(path, "w"), indent=2)
                written.append(path)

    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
