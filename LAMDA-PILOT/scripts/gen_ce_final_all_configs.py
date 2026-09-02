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
  - SketchLoRA: FIXED 2026-09-01 (was wrong from 2026-08-31 through
    2026-09-01 -- the actual bug this file's edit history is about). This
    dict's "sketchlora" entry originally pointed at exps/wave1_final/
    sketchlora_imagenetr_s1993.json, copied verbatim from the OLDER
    ce_eval_wave1_imagenetr20t generator -- the base, non-orth "sketchlora"
    model, from BEFORE the user's "SketchLoRA now means the orthogonal
    variant by default" instruction (given earlier the same session this
    file was first written). That instruction was already correctly applied
    in the separate scripts/gen_sketchlora_ncm_imagenetr_configs.py written
    around the same time -- this file just never got the same fix, so the
    2026-09-01 ce_final_all cluster run trained and CE-profiled the WRONG
    SketchLoRA variant. Its results (run_logs/ce_final_all/ce2/sketchlora/)
    are not usable as "SketchLoRA" and must not be reported as such -- fixed
    below by pointing "sketchlora" at exps/sketchlora_fixedrank_orth05_
    imagenetr_s1993.json (fixed-rank, sketchlora_align_mode="orth", weight
    0.5 -- the same config validated against a real local metrics JSON
    earlier this project and used as "SketchLoRA" in every other current
    comparison table/chart this session). Its model_name is "sketchlora_
    align", not "sketchlora" -- CE2's own output will therefore land under
    run_logs/ce_final_all/ce2/sketchlora_align/, a NEW subfolder, not
    overwrite the old (wrong, base-variant) one; scripts/_bankcap_run_done.py's
    resumability check keys off model_name too, so a resubmission will
    correctly treat this as "not done yet" and actually retrain it, rather
    than skipping on the strength of the old (wrong-model) run.

Base source configs, copied VERBATIM (only prefix and ce2_enabled change):
  seqlora, inflora, treelora, rainbowprompt
      <- exps/wave1_final/<method>_imagenetr_s1993.json
  sketchlora
      <- exps/sketchlora_fixedrank_orth05_imagenetr_s1993.json (fixed
      2026-09-01 -- see the SketchLoRA caveat above; this is NOT under
      exps/wave1_final/, it's the dedicated orth config used everywhere
      else "SketchLoRA" appears this session)
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

# method -> source config path. "sketchlora" fixed 2026-09-01 to point at
# the orth config (see the SketchLoRA caveat above) -- every other entry
# unchanged, still correct.
SOURCES = {
    "olora": "exps/wave1_final_completion/olora_imagenetr_s1993.json",
    "seqlora": "exps/wave1_final/seqlora_imagenetr_s1993.json",
    "inflora": "exps/wave1_final/inflora_imagenetr_s1993.json",
    "treelora": "exps/wave1_final/treelora_imagenetr_s1993.json",
    "sketchlora": "exps/sketchlora_fixedrank_orth05_imagenetr_s1993.json",
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
