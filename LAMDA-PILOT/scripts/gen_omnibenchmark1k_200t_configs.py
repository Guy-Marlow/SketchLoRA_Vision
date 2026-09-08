"""Long-sequence comparison (2026-09-08 user request): all 8 established
methods on a 200-task OmniBenchmark-1k split (vs. the usual 100-task split),
3 seeds each = 24 runs. "How does each method hold up over a much longer
task sequence?"

SPLIT: init_cls=5, increment=5 -> exactly 200 tasks over 1000 classes
(5 + 5*199 = 1000, no remainder) -- half the classes/task of the standing
100-task split (init_cls=10, increment=10), same total class count.

SOURCES: each method's own most-recent/best-established OmniBenchmark-1k
config, verified against the actual files (not assumed) before writing this:
  seqlora, inflora, rainbowprompt -- exps/wave1_final/ (no newer variant
    exists for these three).
  olora, cllora, ease, tuna       -- exps/wave1_final_completion/ (newer
    than plain wave1_final -- olora is the "current-code rerun" per that
    campaign's own docstring; cllora/ease/tuna carry the SETTLED
    post-LR-sweep learning rates: cllora=0.05, ease=1e-2, tuna=5e-3 on
    OmniBenchmark-1k specifically -- see the other-methods-config-and-
    lr-sweep memory note -- confirmed present in these exact files'
    init_lr/later_lr fields before trusting them).
  sketchlora -- exps/sketchlora_orth_omnibenchmark1k/ (the orthogonalized
    fixed-rank variant, this project's standing default "SketchLoRA" per
    the sketchlora-orth-is-now-default-convention -- NOT wave1_final's own
    base/non-orth "sketchlora" config).

Every source config is loaded PER SEED from its own real file (not cloned
from the seed-1993 file with just the seed field swapped) -- preserves
whatever per-seed variation may already exist rather than assuming
uniformity across seeds.

Only dataset-split fields (init_cls/increment) and prefix/sketchlora_diag_dir
(for SketchLoRA, to avoid colliding with the 100-task campaign's own
sketch_diag files) are overridden; every method-specific hyperparameter is
copied verbatim from its source.
"""
import json
import os

OUT_DIR = "exps/omnibenchmark1k_200t"
RUN_LOGS_BASE = "run_logs/omnibenchmark1k_200t"
SEEDS = [1993, 1996, 1999]

# 200 tasks over 1000 classes, no remainder.
SPLIT_OVERRIDES = dict(init_cls=5, increment=5)

SOURCES = {
    "seqlora": "exps/wave1_final/seqlora_omnibenchmark1k_s{seed}.json",
    "inflora": "exps/wave1_final/inflora_omnibenchmark1k_s{seed}.json",
    "rainbowprompt": "exps/wave1_final/rainbowprompt_omnibenchmark1k_s{seed}.json",
    "olora": "exps/wave1_final_completion/olora_omnibenchmark1k_s{seed}.json",
    "cllora": "exps/wave1_final_completion/cllora_omnibenchmark1k_s{seed}.json",
    "ease": "exps/wave1_final_completion/ease_omnibenchmark1k_s{seed}.json",
    "tuna": "exps/wave1_final_completion/tuna_omnibenchmark1k_s{seed}.json",
    "sketchlora": "exps/sketchlora_orth_omnibenchmark1k/sketchlora_orth_omnibenchmark1k_fixedrank_orth05_s{seed}.json",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for method, src_template in SOURCES.items():
        for seed in SEEDS:
            src_path = src_template.format(seed=seed)
            cfg = json.load(open(src_path))
            cfg.update(SPLIT_OVERRIDES)
            cfg["seed"] = [seed]
            cfg["prefix"] = "omnibenchmark1k_200t_{}_s{}".format(method, seed)
            if method == "sketchlora":
                cfg["sketchlora_diag_dir"] = "{}/sketch_diag".format(RUN_LOGS_BASE)
            path = os.path.join(OUT_DIR, "{}_s{}.json".format(method, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} configs under {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
