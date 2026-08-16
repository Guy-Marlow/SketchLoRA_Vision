"""Config generator for the wave1_rerun_treelora campaign (2026-08-16).

TreeLoRA's ORIGINAL wave1_final numbers (exps/wave1_final/treelora_*_s*.json,
run 2026-08-09) predate commit 23c6405 (2026-08-10), which removed TreeLoRA's
dense-slot folding and fixed its fold-specific persistent_state() override --
a real change to how persistent memory is COUNTED (mirrors the O-LoRA
recompute done this session: the fixed fold-buffer term is gone, replaced by
summing what's actually still allocated). Accuracy should be unaffected in
principle (factored vs. folded forward are mathematically identical
outputs), but the wave1_final metrics JSONs were never actually regenerated
under the current code -- this campaign produces one.

Every config is copied VERBATIM from its wave1_final source (same dataset,
task/epoch/LR/reg settings, same seed) with exactly one field changed:
`prefix`, to `wave1_rerun_memfix`. Deliberate, not incidental -- reusing the
original prefix would land this campaign's output at the exact same
run_logs/final/treelora/metrics_*.json path as the stale pre-fix run; on a
RESUMED submission scripts/_bankcap_run_done.py would then see that
already-"done" JSON and skip the rerun entirely. A distinct prefix
guarantees these runs always start "not done" the first time, regardless of
what the original wave1_final run left behind, and leaves the original
stale files untouched for direct before/after comparison.
"""

import json
import os

WAVE1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exps", "wave1_final")

DATASETS = ["cifar224", "food101", "imagenetr", "omnibenchmark1k", "sun397"]
SEEDS = [1993, 1996, 1999]
PREFIX = "wave1_rerun_memfix"


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "exps", "wave1_rerun_treelora")
    os.makedirs(out_dir, exist_ok=True)

    for dataset in DATASETS:
        for seed in SEEDS:
            src = os.path.join(WAVE1, f"treelora_{dataset}_s{seed}.json")
            with open(src) as f:
                cfg = json.load(f)
            assert cfg.get("final_metrics") is True, \
                f"{src} does not have final_metrics=true -- refusing to silently rerun without it"
            cfg["prefix"] = PREFIX
            out = os.path.join(out_dir, f"treelora_{dataset}_s{seed}.json")
            with open(out, "w") as f:
                json.dump(cfg, f, indent=2)
            print(out)


if __name__ == "__main__":
    main()
