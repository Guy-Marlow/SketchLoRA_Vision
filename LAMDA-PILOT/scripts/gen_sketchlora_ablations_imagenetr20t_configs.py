"""Config generator for the SketchLoRA compression-scheme ablation series (user
request 2026-07-31, extended 2026-08-04 with two more variants): ORDINARY
task-incremental training (no boundary_mode key at all -- routes to
trainer.py's plain per-task loop, real task boundaries provided, NOT
bounded_memory/streaming), ImageNet-R full 20-task split, 6 SketchLoRA
variants x 3 seeds (1993/1996/1999 -- not specified by the user, assumed from
this project's standing 3-seed convention; flag if a different seed set was
intended).

Each variant changes exactly ONE axis off the "current"/frozen-v1 baseline
(the config already used throughout round2_slurm_grid/imagenetr_grid), so the
series isolates one mechanism at a time:

  current   -- baseline: adaptive threshold (svd_energy_target=0.01),
               bounded_eviction admission, rank_cap=128, randsvd merge.
               Identical SketchLoRA knobs to every other grid in this
               project; only boundary_mode is stripped to make it an oracle
               run instead of bounded_memory.
  exactsvd  -- same as current, merge_op="exactsvd" instead of "randsvd"
               (isolates: does the randomized-vs-exact projection matter).
  globaleps -- same as current, sketchlora_admission="global_eps" instead of
               "bounded_eviction" (isolates: what does DISABLING the
               eviction-count cap do). bounded_eviction's rule is
               evict = min(residual_total, naive_evict) -- eviction is hard-
               capped at residual_total, i.e. never more than one adapter's
               worth (lora_rank) of directions per merge, which is exactly
               why rank is monotone non-decreasing under it (see prior
               conversation turn's exhaustive verification). global_eps
               drops that cap entirely: r_hat_t = k_eps directly (clamped
               only by rank_cap, confirmed from models/sketchlora.py's own
               comment: "the pure-global-eps branch...can otherwise evict
               far more than that in one merge if the composite's post-fold
               energy spectrum happens to concentrate differently, which is
               the 'retroactive mass-eviction' / post-peak rank collapse
               A5.2 exists to fix"). THIS is "allowing the adapter to
               shrink" -- rank can now decrease between merges, not just
               plateau. rank_cap=128 stays ON: confirmed from the same code
               that global_eps still respects it as a plain clamp
               (`r_hat_t = min(r_hat_t, self.rank_cap)`), independent of the
               admission-rule choice, so this variant isolates the eviction-
               cap axis alone, not rank_cap too. NOT the same thing as
               fd_shrinkage (a separate, unrelated mechanism that shrinks
               the MAGNITUDE of already-kept singular values, FD "pay rent"
               style -- never touched by this variant, stays off/default
               throughout this whole series).
  fixedrank -- svd_energy_target OMITTED (None) -> sketchlora.py's fixed-rank
               path (r_hat_t always = svd_rank = lora_rank = 10, never grows,
               never adapts). sketchlora_admission MUST be "global_eps" here
               (NOT bounded_eviction/floor -- both assert energy_target is
               not None at construction, confirmed against models/
               sketchlora.py's __init__ before writing this). rank_cap=128
               is kept per explicit user instruction ("won't really matter
               either way") -- confirmed structurally inert: fixed mode's
               r_hat_t=10 never approaches a 128 cap, so leaving it set
               changes nothing while keeping the config self-documenting.
  countsketch -- fixedrank's SIBLING, not current's: svd_energy_target
               OMITTED (None), same as fixedrank, REQUIRED here for a
               different reason than fixedrank's own (user correction,
               2026-08-04) -- CountSketch has no singular-value-like
               importance signal of its own to drive an adaptive rank
               threshold the way randsvd/exactsvd's spectrum does. Choosing
               r_hat_t via the SVD-based adaptive-threshold path and then
               discarding that SVD's actual basis in favor of a random hash
               merge would smuggle exact-SVD rank information into a variant
               meant to test hashing WITHOUT any such signal, defeating the
               isolation this whole series exists for. sketchlora_admission
               forced to "global_eps" (only legal choice when energy_target
               is None, same constraint as fixedrank). `cs_rank` intentionally
               omitted -> defaults to svd_rank (10, models/sketchlora.py:284)
               -- matches fixedrank's r_hat_t=10 exactly, so this variant
               isolates the merge-ALGORITHM axis alone (hash-based
               CountSketch vs SVD truncation) at the SAME fixed rank
               fixedrank uses, rather than the merge-ALGORITHM axis at an
               adaptively-chosen rank (which would need that rank to come
               from somewhere -- and nothing legitimate is available).
               rank_cap=128 kept per fixedrank's own precedent; confirmed
               inert here too (neither fixed-rank branch applies it).
  exactsvd_ca -- same as "exactsvd" (current + merge_op=exactsvd), PLUS
               classifier_alignment=True (ca_steps=300, ca_batch=128,
               ca_lr=0.001 -- copied verbatim from the completed local CA
               control run, exps/sketchlora_boltons/100mb_ca.json, the only
               other place this project has run plain CA). Isolates: does
               classifier alignment help/hurt on top of the exact-SVD merge,
               same admission/threshold as current.

final_metrics=True (unlike the bounded_memory grids, where MetricsLogger is
wired in unconditionally): trainer.py's oracle path only constructs
MetricsLogger when this flag is set (trainer.py:139), so it must be turned on
explicitly here to get persistent-memory/FLOP/train-eval-seconds data
comparable to the rest of this project's "final" runs. Note this does NOT
give CE (Ops ledger) data -- utils/ops_ledger.py's OpsLedger/measure_step_macs
are only imported by models/bounded_memory_mixin.py; the oracle path has no
CE-ledger wiring at all currently.
"""
import json
import os

OUT_DIR = "exps/sketchlora_ablations_imagenetr20t"
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(
    dataset="imagenetr",
    memory_size=0, memory_per_class=0, fixed_memory=False, shuffle=True,
    init_cls=10, increment=10, scenario="cil",
    pretrained=True, print_forget=False, final_metrics=True,
    backbone_type="vit_base_patch16_224_lora",
    lora_rank=10, lora_alpha=None,
    batch_size=48, weight_decay=0.0005, min_lr=0.0,
    tuned_epoch=20,
    device=["0"],
    model_name="sketchlora", lora_merge=True, lora_train_merge=True,
    svd_rank=10, svd_oversampling=10,
    lora_n_slots=2, sketch_diag=True,
    sketchlora_lora_wd=0.0,
    init_lr=0.001,
    # ce_profile_every=0 (2026-08-04 correction): disables the measured-CE
    # torch.profiler region tracing entirely (utils/ce_profiler.py's
    # documented safety valve, enabled = ce_profile_every > 0). Without this,
    # trainer.py defaults to 25, but CEProfileController's force_cycles=(0,1)
    # is hardcoded independent of that setting -- tasks 0 and 1 of every one
    # of this series' 20-task runs would still get the SAME whole-task-open
    # profiler session that made ce_smoke_imagenetr5t take ~40min for 4
    # tasks of just the cheapest method. This series doesn't need that
    # region-level breakdown (it validates SketchLoRA's own behavior across
    # variants, not the CE profiler itself), so it stays off. final_metrics
    # (below) is untouched by this -- MetricsLogger and the formula-based
    # ops_ledger (accuracy/persistent-memory/FLOPs/train-eval-seconds, plus
    # the cheap single-batch R2 baseline probe) still run every task exactly
    # as before; only the expensive per-region profiler trace is skipped.
    ce_profile_every=0,
    # no boundary_mode key -- ordinary per-task oracle loop (trainer.py's
    # default path when boundary_mode is absent), real task boundaries.
)

VARIANT_CFG = {
    "current": dict(
        merge_op="randsvd", svd_energy_target=0.01,
        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        sketchlora_eviction_reading="conformant",
    ),
    "exactsvd": dict(
        merge_op="exactsvd", svd_energy_target=0.01,
        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        sketchlora_eviction_reading="conformant",
    ),
    "globaleps": dict(
        merge_op="randsvd", svd_energy_target=0.01,
        sketchlora_admission="global_eps", sketchlora_rank_cap=128,
        # sketchlora_eviction_reading intentionally absent -- that key is only
        # ever read inside the bounded_eviction branch of _compress(); it has
        # no effect under global_eps and is omitted rather than set-but-inert.
    ),
    "fixedrank": dict(
        merge_op="randsvd",
        # svd_energy_target intentionally absent -> None -> fixed-rank path
        sketchlora_admission="global_eps",   # only legal choice when energy_target is None
        sketchlora_rank_cap=128,             # kept per user instruction; inert here (r_hat_t always 10)
    ),
    "exactsvd_ca": dict(
        merge_op="exactsvd", svd_energy_target=0.01,
        sketchlora_admission="bounded_eviction", sketchlora_rank_cap=128,
        sketchlora_eviction_reading="conformant",
        classifier_alignment=True, ca_steps=300, ca_batch=128, ca_lr=0.001,
    ),
    "countsketch": dict(
        merge_op="countsketch",
        # svd_energy_target intentionally absent -> None -> fixed-rank path,
        # REQUIRED (not just fixedrank's convenience): CountSketch has no
        # singular-value-like signal of its own to drive an adaptive
        # threshold, so there is no legitimate rank source other than a
        # fixed one here -- see module docstring.
        sketchlora_admission="global_eps",   # only legal choice when energy_target is None
        sketchlora_rank_cap=128,             # kept per fixedrank's own precedent; inert here too
        # cs_rank intentionally omitted -> defaults to svd_rank (10),
        # matching fixedrank's r_hat_t=10 exactly.
    ),
}

# order matters -- variant-major execution order in the .slurm script; the two
# new variants are appended after the original four, per explicit instruction
# ("after the current run order"), not interleaved with them.
VARIANTS = ["current", "exactsvd", "globaleps", "fixedrank", "exactsvd_ca", "countsketch"]
SEEDS = [1993, 1996, 1999]

configs = []
for variant in VARIANTS:
    for seed in SEEDS:
        cfg = dict(BASE)
        cfg.update(VARIANT_CFG[variant])
        cfg["seed"] = [seed]
        cfg["prefix"] = "sketchlora_ablations_imagenetr20t_{}_s{}".format(variant, seed)
        fname = "sketchlora_{}_s{}.json".format(variant, seed)
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        configs.append(fname)

print("wrote {} configs to {}".format(len(configs), OUT_DIR))
for c in configs:
    print(c)
