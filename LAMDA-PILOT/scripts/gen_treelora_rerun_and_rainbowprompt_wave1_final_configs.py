"""Configs for the 2026-08-12 wave1_final follow-up (user request): (a) a
TreeLoRA RERUN and (b) a NEW RainbowPrompt run, both folded into the same
wave1_final results family.

(a) TreeLoRA: this generator writes NOTHING for TreeLoRA. Its bug fix
(models/treelora.py, this session -- removed the per-task adapter bank the
reference implementation doesn't have, see the treelora_adapter_bank_removed
memory) is code-only, not config-only, so the EXISTING
exps/wave1_final/treelora_{dataset}_s{seed}.json (all 15, written by
gen_wave1_final_configs.py, confirmed present and untouched) are reused
byte-for-byte. The rerun exists purely to regenerate results under corrected
code; "same config" was explicit in the user's request.

(b) RainbowPrompt: NEW to the wave1_final family (wasn't part of the original
5-method grid). Architecture hyperparameters (rp_length, self_attn_idx,
KI_iter, pull_constraint_coeff, use_linear, rp_D1, rp_D2, batch_size,
weight_decay=0.0, min_lr=0.0, tuned_epoch=20) copied verbatim from
scripts/gen_cllora_ease_tuna_rainbow_lr_refine_configs.py's own
METHOD_CFG["rainbowprompt"] block -- nothing about the architecture/training
loop changes here. What DOES change from that LR-refine sweep:
  - init_lr fixed at 1e-3 (not swept) -- the value that sweep's own results
    confirmed cleanly across all 5 datasets as RainbowPrompt's best LR.
  - NO stop_after_tasks -- full, untruncated task counts this time
    (cifar224/imagenetr/sun397/food101/omnibenchmark1k), matching every other
    wave1_final method.
  - final_metrics=True, print_forget=True (the LR-refine sweep had both off,
    since it only needed a final top-1/top5 number for a short probe) -- this
    run is meant to sit alongside the other 5 methods' wave1_final data, so it
    needs the same persistent-memory/inference-FLOPs/CE-ledger accounting
    they all have.
  - prefix/filename convention matches wave1_final's own
    ("wave1_final_rainbowprompt_{dataset}_s{seed}" / "rainbowprompt_{dataset}_
    s{seed}.json"), written into exps/wave1_final/ itself (not a separate
    directory) -- this is an extension of that campaign, not a new one.

Dataset splits (init_cls/increment) and seeds are copied from
gen_wave1_final_configs.py's own DATASET_CFG/SEEDS -- identical values, kept
as a separate literal here rather than importing that module, since the two
generators are meant to stay independently readable.
"""
import json
import os

OUT_DIR = "exps/wave1_final"

RAINBOWPROMPT_CFG = dict(
    model_name="rainbowprompt", backbone_type="vit_base_patch16_224_rainbowprompt",
    tuned_epoch=20, batch_size=48, init_lr=0.001, weight_decay=0.0, min_lr=0.0,
    rp_length=20, self_attn_idx=[0, 1, 2, 3, 4, 5], KI_iter=10,
    pull_constraint_coeff=0.1, use_linear=True, rp_D1=28, rp_D2=56,
)

DATASET_CFG = {
    "cifar224": dict(init_cls=10, increment=10),
    "imagenetr": dict(init_cls=10, increment=10),
    "food101": dict(init_cls=6, increment=5),
    "sun397": dict(init_cls=37, increment=40),
    "omnibenchmark1k": dict(init_cls=10, increment=10),
}

SEEDS = [1993, 1996, 1999]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for dataset, dcfg in DATASET_CFG.items():
        for seed in SEEDS:
            cfg = dict(
                dataset=dataset,
                memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
                init_cls=dcfg["init_cls"], increment=dcfg["increment"],
                seed=[seed], scenario="cil",
                pretrained=True, print_forget=True, final_metrics=True,
                tuned_epoch=RAINBOWPROMPT_CFG["tuned_epoch"],
                batch_size=RAINBOWPROMPT_CFG["batch_size"],
                init_lr=RAINBOWPROMPT_CFG["init_lr"],
                weight_decay=RAINBOWPROMPT_CFG["weight_decay"],
                min_lr=RAINBOWPROMPT_CFG["min_lr"],
                model_name=RAINBOWPROMPT_CFG["model_name"],
                backbone_type=RAINBOWPROMPT_CFG["backbone_type"],
                device=["0"],
                prefix="wave1_final_rainbowprompt_{}_s{}".format(dataset, seed),
            )
            for k in ("rp_length", "self_attn_idx", "KI_iter",
                      "pull_constraint_coeff", "use_linear", "rp_D1", "rp_D2"):
                cfg[k] = RAINBOWPROMPT_CFG[k]

            path = os.path.join(OUT_DIR, "rainbowprompt_{}_s{}.json".format(dataset, seed))
            json.dump(cfg, open(path, "w"), indent=2)
            written.append(path)
    print("wrote {} RainbowPrompt configs to {}/".format(len(written), OUT_DIR))


if __name__ == "__main__":
    main()
