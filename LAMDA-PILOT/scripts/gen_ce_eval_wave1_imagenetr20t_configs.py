"""CE (computational efficiency) evaluation campaign (2026-08-20 user
request): single-seed ImageNet-R-20t run for each of the 9 wave-1 methods,
with full CE/CE2 logging enabled, to conclusively compare real MAC cost
across methods over a 20-task horizon.

Follows a 9-agent CE-logging audit (one agent per method). PROJECT POLICY
(2026-08-20, explicit user directive): **CE1 (the legacy `OpsLedger`/narrow
`ce_region`-tag system) is obsolete and must never be used, in this
campaign or any future one. CE2 (`ce2_enabled=true`) is the only relevant
CE metric.** Every config below sets `ce2_enabled=true` accordingly.
Verified directly against `trainer.py:144,197-198,214-215` (not just
inferred from the audit): `ce_ledger` is only ever constructed when
`hasattr(model, "_train_adapter") and not args.get("ce2_enabled")`, so
`ce2_enabled=true` guarantees CE1 is never constructed for ANY method,
including the 5 that would otherwise qualify for it (O-LoRA/SeqLoRA/
InfLoRA/TreeLoRA/SketchLoRA) -- this is by construction, not a side effect
of method choice. CL-LoRA/TUNA/EASE/RainbowPrompt have no `_train_adapter`
and so were NEVER eligible for CE1 regardless; CE2 is their only CE source
either way. `final_metrics`/`print_forget` (the standard wave-1 accuracy/
memory logging, unrelated to which CE system is active) are carried over
verbatim from each method's source config, unchanged.

KNOWN CAVEATS ON THE RESULTING NUMBERS (do not silently drop these when
reporting results from this campaign):
  - TUNA: CE2's boundary window opens AFTER `merge()`'s real EMR task-vector
    construction already ran (models/tuna.py) -- that cost lands in TUNA's
    "train" bucket, not "boundary". Total (train+eval+boundary) is still
    correct; the train/boundary SPLIT is not.
  - RainbowPrompt: no `ce2_boundary()` call site exists in
    models/rainbowprompt.py at all -- its boundary bucket will read exactly
    zero regardless of what add_task_slot() actually costs (a small,
    genuinely low-cost op, so low practical impact, but the zero is not
    informative).
  - O-LoRA: the one known CE gap (a stale streaming-mode formula) does not
    apply to this campaign (oracle CIL only, not bounded-memory streaming).
  - InfLoRA: `fold_up_to`'s actual fold matmul (~141M MACs/task) is
    genuinely untagged/unformulated in both CE1 and CE2 -- a small,
    confirmed-scope gap, not fixed here.
  - SketchLoRA: uses "sketchlora" (the base, canonical wave1 method), NOT
    the newer sketchlora_align/sketchlora_orth variants -- the CE-invisible-
    regularizer bug found in models/sketchlora_align.py does not apply to
    this campaign's config.

Base source configs, copied VERBATIM (only prefix and ce2_enabled change):
  olora, seqlora, inflora, treelora, sketchlora, rainbowprompt
      <- exps/wave1_final/<method>_imagenetr_s1993.json
  cllora, tuna, ease
      <- exps/wave1_final_completion/<method>_imagenetr_s1993.json
      (settled learning rates; wave1_final itself never included these 3)

Single seed (1993) per explicit user request -- this campaign answers "how
expensive is each method," not a statistical accuracy comparison, so one
seed is sufficient and keeps this a single pass per method.
"""
import copy
import json
import os

OUT_DIR = "exps/ce_eval_wave1_imagenetr20t"
SEED = 1993

# method -> source config path
SOURCES = {
    "olora": "exps/wave1_final/olora_imagenetr_s1993.json",
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
        cfg["prefix"] = "ce_eval_wave1_imagenetr20t_{}_s{}".format(method, SEED)
        out = os.path.join(OUT_DIR, "{}_s{}.json".format(method, SEED))
        json.dump(cfg, open(out, "w"), indent=2)
        written.append(out)
    print("wrote {} configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
