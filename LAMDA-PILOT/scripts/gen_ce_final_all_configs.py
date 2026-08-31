"""CE (computational efficiency) FINAL evaluation campaign (2026-08-31 user
request): single-seed ImageNet-R-20t run for each of the 9 wave-1 methods,
with full CE2 logging enabled -- a from-scratch replacement for
ce_eval_wave1_imagenetr20t, not a continuation of it.

WHY A NEW CAMPAIGN INSTEAD OF RESUBMITTING THE OLD ONE (the actual reason
this file exists): ce_eval_wave1_imagenetr20t was first run 2026-08-22 and
resubmitted 2026-08-30/31 to pick up its 4 OOM-killed methods (see that
script's own HOST MEMORY note). But its resumable skip logic
(scripts/_bankcap_run_done.py) only checks "does a metrics file marked
status==done exist" -- it has zero awareness of whether the underlying
model code has changed since that file was written. The 5 methods that
"completed" (SeqLoRA/InfLoRA/TreeLoRA/SketchLoRA/CL-LoRA) all did so on
2026-08-22, and were silently SKIPPED (not re-run) on the 2026-08-30
resubmission purely because their old metrics files were already marked
done -- even though 9 days of further project work happened in between
(including the sketchlora_align/orthref/retain-admission additions on
2026-08-23, one day later). User-confirmed 2026-08-31: none of those 5
results are recent; all of them are that same original 2026-08-22 run,
never refreshed. A resubmission of the OLD campaign would perpetuate this
-- its skip logic has no way to tell "old but technically done" apart from
"actually current." A fresh campaign name/prefix sidesteps the problem
entirely: nothing here can accidentally match an old metrics file, so every
one of the 9 methods genuinely (re)trains from current code this time.

PROJECT POLICY (unchanged from ce_eval_wave1_imagenetr20t): CE1 (legacy
OpsLedger) is obsolete and must never be used -- CE2 (`ce2_enabled=true`)
is the only relevant CE metric, for every method. Verified directly against
trainer.py:144,197-198,214-215 that ce2_enabled=true guarantees CE1 is
never constructed for ANY method. final_metrics/print_forget (the standard
accuracy/memory logging, unrelated to which CE system is active) are
carried over verbatim from each method's source config, unchanged.

KNOWN CAVEATS ON THE RESULTING NUMBERS -- carried over from
ce_eval_wave1_imagenetr20t's own docstring, still true here (do not
silently drop these when reporting results from this campaign):
  - TUNA: CE2's boundary window opens AFTER `merge()`'s real EMR task-vector
    construction already ran (models/tuna.py) -- that cost lands in TUNA's
    "train" bucket, not "boundary". Total (train+eval+boundary) is still
    correct; the train/boundary SPLIT is not.
  - RainbowPrompt: no `ce2_boundary()` call site exists in
    models/rainbowprompt.py at all -- its boundary bucket will read exactly
    zero regardless of what add_task_slot() actually costs.
  - InfLoRA: `fold_up_to`'s actual fold matmul (~141M MACs/task) is
    genuinely untagged/unformulated in both CE1 and CE2.
  - SketchLoRA: uses "sketchlora" (the base, canonical wave1 method), NOT
    the newer sketchlora_align/sketchlora_orth variants.

Base source configs, copied VERBATIM (only prefix and ce2_enabled change) --
same sources as ce_eval_wave1_imagenetr20t, re-checked 2026-08-31 against
git history since (nothing newer exists for any of these 9 methods):
  seqlora, inflora, treelora, sketchlora, rainbowprompt
      <- exps/wave1_final/<method>_imagenetr_s1993.json
  olora, cllora, tuna, ease
      <- exps/wave1_final_completion/<method>_imagenetr_s1993.json
      (O-LoRA: current-code rerun, commit 23c6405. cllora/tuna/ease:
      settled learning rates; wave1_final itself never included these 3.)

Single seed (1993) per explicit user request -- this campaign answers "how
expensive is each method," not a statistical accuracy comparison.
"""
import copy
import json
import os

OUT_DIR = "exps/ce_final_all"
SEED = 1993

# method -> source config path (unchanged from ce_eval_wave1_imagenetr20t --
# re-verified 2026-08-31 that nothing newer exists for any of these 9)
SOURCES = {
    "olora": "exps/wave1_final_completion/olora_imagenetr_s1993.json",
    "seqlora": "exps/wave1_final/seqlora_imagenetr_s1993.json",
    "inflora": "exps/wave1_final/inflora_imagenetr_s1993.json",
    "treelora": "exps/wave1_final/treelora_imagenetr_s1993.json",
    "sketchlora": "exps/wave1_final/sketchlora_imagenetr_s1993.json",
    "rainbowprompt": "exps/wave1_final/rainbowprompt_imagenetr_s1993.json",
    "cllora": "exps/wave1_final_completion/cllora_imagenetr_s1993.json",
    "tuna": "exps/wave1_final_completion/tuna_imagenetr_s1993.json",
    "ease": "exps/wave1_final_completion/ease_imagenetr_s1993.json",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for method, src in SOURCES.items():
        cfg = copy.deepcopy(json.load(open(src)))
        assert cfg.get("final_metrics") is True, \
            "{} does not have final_metrics=true -- CE reporting needs it".format(src)
        cfg["ce2_enabled"] = True
        cfg["prefix"] = "ce_final_all_{}_s{}".format(method, SEED)
        out = os.path.join(OUT_DIR, "{}_s{}.json".format(method, SEED))
        json.dump(cfg, open(out, "w"), indent=2)
        written.append(out)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
