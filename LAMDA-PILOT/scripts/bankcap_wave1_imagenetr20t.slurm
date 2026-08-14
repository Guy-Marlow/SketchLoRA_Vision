#!/bin/bash
#SBATCH --job-name=bankcap_wave1_imagenetr20t
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bankcap_wave1_imagenetr20t/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bankcap_wave1_imagenetr20t/slurm_%j.err
#
# Bounded-bank-memory sensitivity campaign, ImageNet-R-20t, oracle CIL, 3 seeds
# (1993/1996/1999) per run, 2026-08-14 user request. Answers: does SketchLoRA
# make better use of a fixed memory budget than methods retrofitted with the
# new bank_cap_mb setting (docs/bounded_bank_memory_changelog.md)?
#
# EXCLUDED, both by explicit user decision this session:
#   - InfLoRA: its persistent floor (frozen_delta_q/v, ~54MB, always allocated
#     from construction regardless of budget) dominates any cap in the range
#     this study cares about -- no useful middle ground between "always
#     binds" and "never binds", unlike the other six methods.
#   - TUNA: no settled/verified learning rate for ImageNet-R yet (the
#     cllora_ease_tuna_lr_refine2 sweep has only reached Food-101 for TUNA as
#     of 2026-08-14).
#
# TWO PHASES, each structured as SketchLoRA-first-then-derived-cap:
#
#   Phase "rank16": SketchLoRA on ImageNet-R-20t, wave1_final settings
#     UNCHANGED except sketchlora_rank_cap: 128 -> 16 (wave1_final SketchLoRA
#     already runs in ADAPTIVE mode -- svd_energy_target=0.01,
#     sketchlora_admission=bounded_eviction -- with sketchlora_rank_cap as an
#     existing ceiling knob; this phase just tightens that ceiling, nothing
#     else about the method's mode changes). 3 seeds. Once all 3 finish, take
#     the MEAN of their final-task persistent_mb (from final_metrics json --
#     genuinely seed-dependent here, unlike the other 6 methods' deterministic
#     footprint, since adaptive rank selection depends on the actual singular
#     spectra encountered), round to the nearest whole MB -> that becomes
#     bank_cap_mb for every other method this phase.
#   Then: O-LoRA, TreeLoRA, RainbowPrompt, CL-LoRA, EASE (in that order) x 3
#     seeds each, ImageNet-R-20t, wave1_final settings + bank_cap_mb=<derived>.
#
#   Phase "rank32": identical structure, sketchlora_rank_cap=32 instead of 16
#     (yielding a larger derived cap for the second half).
#
# CL-LoRA/EASE HAVE NO wave1_final ImageNet-R CONFIG (never part of the
# original wave1_final grid) -- scripts/gen_bankcap_wave1_imagenetr20t_
# configs.py builds full-length (20-task, 20-epoch, final_metrics=true)
# configs from scratch for these two, using the LRs settled by the
# cllora_ease_tuna_lr_refine2 sweep as of 2026-08-14: CL-LoRA locked at 0.05
# for ImageNet-R (matches Food-101's separately-locked value); EASE settled
# at 1e-2 uniformly (tied-for-best on Food-101/ImageNet-R in that sweep). See
# that script's own docstring for the full field-by-field provenance -- this
# is a genuine judgment call, not pulled from an existing "official" config,
# flagged to the user when this script was written.
#
# WALL TIME (120h, generous but not certainly sufficient): O-LoRA/TreeLoRA/
# SketchLoRA measured at ~30min/seed on this exact split (wave1_final logs,
# 2026-08-06/07). CL-LoRA/EASE are NOT wave1_final-measured on ImageNet-R --
# extrapolated from this session's 2-epoch/3-task smoke-test rate (~7min) to
# full 20-epoch/20-task scale (~66x) gives ~8h/seed, i.e. ~24h per method
# across 3 seeds. Rough total: phase = ~1.5h (sketchlora) + ~1.5h (olora+
# treelora+rainbowprompt) + ~48h (cllora+ease) =~ 51h; both phases =~ 102h.
# The whole campaign is resumable end-to-end (see run_one below), so a SLURM
# time-limit kill is not fatal -- just resubmit, already-finished runs are
# skipped and the derived-cap computation re-derives the identical value from
# the now-complete metrics files.
#
# RESUMABLE BY CONSTRUCTION, no separate checkpoint state needed: run_one()
# skips any config whose final_metrics JSON already shows status=="done"
# (via scripts/_bankcap_run_done.py -- deliberately NOT a grep of the .out
# log's "Average Accuracy (CNN):" line, which trainer.py prints once per
# TASK, not once per completed run; using that as the completion signal
# would falsely mark a run killed by a wall-time cutoff after its first
# task as "done" forever on resubmission, while its metrics JSON stays
# stuck below status=="done" forever -- exactly what the derived-cap
# computation below needs, so the two checks must never disagree). The
# derived-cap computation only requires the 3 sketchlora metrics files to
# exist with status:"done" (already true once those 3 runs are skip-fast on
# a resume) and is a pure deterministic function of their contents, so
# recomputing it on a resume reproduces the exact same bank_cap_mb and
# therefore the exact same regenerated configs for the other 5 methods.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=bankcap_wave1_imagenetr20t
CONFIG_ROOT="exps/$CAMPAIGN"
OUT_ROOT="run_logs/$CAMPAIGN"
mkdir -p "$CONFIG_ROOT" "$OUT_ROOT"

LOCKFILE="$OUT_ROOT/.$CAMPAIGN.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another $CAMPAIGN.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Pre-flight sanity check -- single-GPU by construction, sequential loop,
# same idiom as every other script in this project.
PREFLIGHT_ERR="$(mktemp)"
NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>"$PREFLIGHT_ERR")
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat "$PREFLIGHT_ERR"
  rm -f "$PREFLIGHT_ERR"
  exit 1
fi
rm -f "$PREFLIGHT_ERR"
if [ "$NGPU" != "1" ]; then
  echo "NOTE: torch.cuda.device_count()=${NGPU} (expected 1 for this job's --gres=gpu:1) -- proceeding on CUDA_VISIBLE_DEVICES=0 regardless."
fi
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

run_one() {
  # $1 = path to config json. Dedups by config basename -- see
  # gen_bankcap_wave1_imagenetr20t_configs.py's docstring for why filenames
  # are phase-unique (rank16 vs rank32 must never share a basename here).
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local outfile="${OUT_ROOT}/${tag}.out"

  # Completion check: the final_metrics JSON's own status=="done" (every
  # config in this campaign sets final_metrics=true) via
  # scripts/_bankcap_run_done.py -- NOT a grep of the .out log's "Average
  # Accuracy (CNN):" line. That line prints once per TASK, not once per
  # completed run, so on a run killed by a wall-time cutoff after its first
  # task, it would already be present -- run_one would then skip that run
  # forever on every resubmission while its metrics JSON stays stuck below
  # status=="done" forever, which is exactly the signal
  # bankcap_mean_from_metrics.py needs to compute the derived cap: a
  # permanent deadlock, not silent corruption, but stuck either way. Using
  # the identical status=="done" check in both places rules this out by
  # construction -- run_one and the cap computation can never disagree
  # about whether a run is actually finished.
  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

run_phase() {
  # $1 = phase tag (e.g. "rank16"), $2 = sketchlora_rank_cap for this phase
  local phase="$1" rankcap="$2"
  echo "======== PHASE ${phase} (sketchlora_rank_cap=${rankcap}) START $(date) ========"

  # -- stage A: SketchLoRA, 3 seeds, sketchlora_rank_cap=$rankcap -----------
  local sk_dir="${CONFIG_ROOT}/${phase}_sketchlora"
  echo "---- generating stage-A (sketchlora) configs: ${phase} ----"
  python scripts/gen_bankcap_wave1_imagenetr20t_configs.py \
    --sketchlora-rank-cap "$rankcap" --phase-tag "$phase" --out-dir "$sk_dir"
  for cfg in "$sk_dir"/*.json; do
    run_one "$cfg"
  done

  # -- derive this phase's bank_cap_mb from the 3 just-finished runs -------
  local metrics_glob="run_logs/final/sketchlora/metrics_sketchlora_imagenetr_bankcap_${phase}_s*.json"
  local capmb
  capmb="$(python scripts/bankcap_mean_from_metrics.py --glob "$metrics_glob")"
  if [ -z "$capmb" ]; then
    echo "FATAL: derived bank_cap_mb computation for phase ${phase} produced no output -- aborting campaign (see stderr above for the reason)."
    exit 1
  fi
  echo "==== PHASE ${phase}: derived bank_cap_mb = ${capmb} (mean final-task persistent_mb over 3 sketchlora seeds at rank_cap=${rankcap}, rounded to nearest whole MB) $(date) ===="

  # -- stage B: O-LoRA, TreeLoRA, RainbowPrompt, CL-LoRA, EASE, 3 seeds each,
  # bank_cap_mb=$capmb, in that explicit order -----------------------------
  local other_tag="${phase}_capderived"
  local other_dir="${CONFIG_ROOT}/${other_tag}"
  echo "---- generating stage-B (bank_cap_mb=${capmb}) configs: ${other_tag} ----"
  python scripts/gen_bankcap_wave1_imagenetr20t_configs.py \
    --bank-cap-mb "$capmb" --phase-tag "$other_tag" --out-dir "$other_dir"
  for method in olora treelora rainbowprompt cllora ease; do
    for cfg in "$other_dir/${method}_${other_tag}"_s*.json; do
      run_one "$cfg"
    done
  done

  echo "======== PHASE ${phase} COMPLETE $(date) ========"
}

echo "==== ${CAMPAIGN} start: 2 phases (sketchlora rank_cap 16, then 32), 6 methods/phase (sketchlora + 5 derived-cap), 3 seeds each = 36 total runs $(date) ===="

run_phase rank16 16
run_phase rank32 32

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE $(date) ===="
