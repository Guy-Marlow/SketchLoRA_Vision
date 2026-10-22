#!/bin/bash
#SBATCH --job-name=sketchlora_sens_ablations_wave2
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_sens_ablations_wave2/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_sens_ablations_wave2/slurm_%j.err
#
# SketchLoRA sensitivity + ablation campaign, wave 2 (2026-10-22 user
# request). 120 new runs total (scripts/gen_sketchlora_sens_ablations_wave2_
# configs.py's own docstring has the exact breakdown) across four axes, 3
# seeds each (1993/1996/1999):
#   1. orth-weight sensitivity (align_weight in 0.1/0.3/0.5/0.7/0.9) on
#      SUN397 and Food-101.
#   2. sketch-rank sensitivity (svd_rank in 5/10/15/20/25) on SUN397 and
#      Food-101 -- shares its (0.5, 10) baseline cell with axis 1, which is
#      REUSED from the existing master-table configs
#      (exps/sketchlora_orth_wave1_datasets/{sun397,food101}_s<seed>.json),
#      not regenerated or rerun here.
#   3. merge-op comparison (exactsvd / randsvd+random-rank-selection /
#      freqdir "FD Separate" / freqdir_dense "FD Sum" / countsketch, each at
#      the baseline orth regularizer) on OmniBenchmark-1k, SUN397, Food-101.
#   4. align-mode ablation (without_orth / aligned_a / orth_a_aligned_b) on
#      OmniBenchmark-1k, SUN397, Food-101.
#
# 168h (1 week) wall time per explicit user instruction -- OmniBenchmark-1k's
# 100-task split alone contributes 24 of these runs (merge-op + align-
# ablation axes, 3 seeds each) at ~3h/run per this project's other
# OmniBenchmark-1k SketchLoRA campaigns, and SUN397/Food-101 contribute the
# other 96 runs at shorter per-run wall time. Almost certainly will NOT
# finish in one submission -- relies on the RESUMABLE design below to make
# partial progress across repeated resubmissions, same as every other long
# campaign here, not a bug, expected.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as
# scripts/sketchlora_orth_sens_ablations.slurm / sketchlora_orth_sens_
# omnibenchmark1k.slurm (this cluster has exactly one GPU available).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align and this campaign's own prefix, so it
# can only ever match a genuine attempt from THIS campaign).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's log
# -- run_one() below backs up any existing outfile first (same convention as
# every other campaign script here).
#
# ALL OUTPUT under run_logs/sketchlora_sens_ablations_wave2/ (.out logs here,
# split into sensitivity/<dataset>, mergeop/<dataset>, align_ablation/
# <dataset> subdirs matching the config layout; sketch_diag under its own
# diag/ subdir per category/dataset -- see the config generator;
# final_metrics land at the usual project-wide run_logs/final/ location, same
# as every other oracle-path campaign -- ce2_enabled is NOT set here, this
# campaign is about accuracy/forgetting/rank behavior, not CE profiling).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_sens_ablations_wave2
CONFIG_DIR="exps/$CAMPAIGN"
OUT_DIR="run_logs/$CAMPAIGN"
mkdir -p "$CONFIG_DIR" "$OUT_DIR"

LOCKFILE="$OUT_DIR/.$CAMPAIGN.lock"
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
# same idiom as scripts/sketchlora_orth_sens_ablations.slurm.
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

echo "==== config generation start $(date) ===="
python scripts/gen_sketchlora_sens_ablations_wave2_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sensitivity/sun397
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${subdir}/${tag} $(date) ===="
    return
  fi
  if [ -f "$outfile" ]; then
    mv "$outfile" "${outfile}.prev-${SLURM_JOB_ID:-unknown}"
  fi
  echo "==== START ${subdir}/${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${subdir}/${tag} $(date) ===="
}

echo "==== ${CAMPAIGN} start: 120 runs, single GPU $(date) ===="

while IFS= read -r config; do
  run_one "$config"
done < <(find "$CONFIG_DIR" -name "*.json" | sort)

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
