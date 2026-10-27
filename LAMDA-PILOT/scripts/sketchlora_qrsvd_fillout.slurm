#!/bin/bash
#SBATCH --job-name=sketchlora_qrsvd_fillout
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_qrsvd_fillout/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_qrsvd_fillout/slurm_%j.err
#
# SketchLoRA QR-decomposition (merge_op="qrsvd") fill-out campaign
# (2026-10-27 user request). 168 runs total (scripts/gen_sketchlora_qrsvd_
# fillout_configs.py's own docstring has the exact breakdown) across four
# categories, 3 seeds each (1993/1996/1999):
#   1. main5 (15 runs) -- the 5 master-table benchmarks (CIFAR-100-10t,
#      ImageNet-R-20t, SUN397-10t, Food-101-20t, OmniBenchmark-1k-100t) at
#      the baseline cell (align_weight=0.5, svd_rank=10), mirroring the
#      existing RandomSVD master-table configs with merge_op swapped to
#      "qrsvd".
#   2. omni200t (3 runs) -- OmniBenchmark-1k's 200-task split, same
#      baseline cell, merge_op="qrsvd". 200 sequential tasks x 3 seeds --
#      the single most expensive category here.
#   3. sensitivity_lambda (75 runs) -- align_weight in
#      {0.1,0.3,0.5,0.7,0.9}, svd_rank=10, on all 5 datasets.
#   4. sensitivity_rank (75 runs) -- svd_rank in {5,10,15,20,25},
#      align_weight=0.5, on all 5 datasets.
# Categories 3+4 are intentionally NOT deduplicated against category 1's
# own baseline cell (see the config generator's docstring) -- matches the
# user's own explicit run-count arithmetic.
#
# 168h (1 week) wall time per explicit user instruction -- OmniBenchmark-
# 1k-200t (3 runs, 200 tasks each) and OmniBenchmark-1k-100t's 33 total
# runs (3 main5 + 30 sensitivity) are the dominant wall-time cost, exactly
# as the user flagged. Almost certainly will NOT finish in one submission
# -- relies on the RESUMABLE design below to make partial progress across
# repeated resubmissions, same as every other long campaign here, not a
# bug, expected.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every
# other campaign script in this project (this cluster has exactly one GPU
# available).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align and this campaign's own prefix, so
# it can only ever match a genuine attempt from THIS campaign).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's log
# -- run_one() below backs up any existing outfile first (same convention
# as every other campaign script here).
#
# ALL OUTPUT under run_logs/sketchlora_qrsvd_fillout/ (.out logs here,
# split into main5/<dataset>, omni200t/omnibenchmark1k_200t,
# sensitivity_lambda/<dataset>, sensitivity_rank/<dataset> subdirs matching
# the config layout; sketch_diag under its own diag/ subdir per category/
# dataset -- see the config generator; final_metrics land at the usual
# project-wide run_logs/final/ location, same as every other oracle-path
# campaign -- ce2_enabled is NOT set here, matching the RandomSVD configs
# this campaign fills out, which never had it either).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_qrsvd_fillout
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
# same idiom as every other campaign script in this project.
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
python scripts/gen_sketchlora_qrsvd_fillout_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sensitivity_lambda/sun397
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

# Run order: main5 and omni200t first (the master-table-equivalent
# baselines, most likely to be wanted first for any partial-progress
# check), then sensitivity_lambda, then sensitivity_rank. `sort` within
# each category keeps datasets/variants/seeds deterministic.
echo "==== ${CAMPAIGN} start: 168 runs, single GPU $(date) ===="

for category in main5 omni200t sensitivity_lambda sensitivity_rank; do
  while IFS= read -r config; do
    run_one "$config"
  done < <(find "$CONFIG_DIR/$category" -name "*.json" 2>/dev/null | sort)
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
