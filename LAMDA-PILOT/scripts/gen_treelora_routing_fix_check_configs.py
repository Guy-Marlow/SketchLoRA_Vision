"""Config generator for the treelora_routing_fix_check diagnostic (2026-08-17).

Verifies the utils/kd_tree.py insert_grad fix (frac = depth / total_rounds,
replicating the reference's actual x-lora_depth accumulation instead of the
previously-implemented reduced single-accumulation form -- see that file's
module docstring for why the reduced form was not actually inert for
tree_search/_update_similarity's routing decision, only for get_loss's
self-normalized regularizer-loss magnitude) against real training dynamics,
on this local single-GPU machine.

Both configs are VERBATIM copies of exps/wave1_final/treelora_imagenetr_s1993.json
(same rank/batch/lr/epoch/weight_decay convention, same seed) with two fields
changed: `stop_after_tasks: 10` (truncates the run to the first 10 of
ImageNet-R's 20 standard 10-class tasks -- trainer.py's own supported
truncation mechanism, same convention as exps/ce_smoke_imagenetr5t and
exps/cllora_ease_tuna_lr_refine2 -- chosen specifically so this run's
checkpoints are directly comparable, task-for-task, against the first 10
checkpoints of the already-complete wave1_rerun_treelora imagenetr_s1993 run,
rather than defining a differently-shaped 10-task split), and `prefix`
(distinct per reg value, so neither collides with wave1_final/
wave1_rerun_treelora's own metrics paths).

reg=0.1 is the current, already-audited production value (config A -- what
this campaign's earlier runs actually used). reg=0.5 is the reference repo's
own NLP/TRACE launch-script default (config B -- generated here for
convenience but not necessarily run; only launched if config A's trajectory
doesn't clearly improve on the routing fix alone).
"""

import json
import os

WAVE1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exps", "wave1_final")
SRC = os.path.join(WAVE1, "treelora_imagenetr_s1993.json")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "exps", "treelora_routing_fix_check")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SRC) as f:
        base = json.load(f)
    assert base.get("final_metrics") is True

    for reg, tag in [(0.1, "reg01"), (0.5, "reg05")]:
        cfg = dict(base)
        cfg["stop_after_tasks"] = 10
        cfg["reg"] = reg
        cfg["prefix"] = f"treelora_routing_fix_check_{tag}_imagenetr10t_s1993"
        out = os.path.join(OUT_DIR, f"treelora_imagenetr10t_{tag}_s1993.json")
        with open(out, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print(out)


if __name__ == "__main__":
    main()
