#!/bin/bash
#SBATCH --job-name=olora_sketch_fd_divergence
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/olora_sketch_fd_divergence/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/olora_sketch_fd_divergence/slurm_%j.err
#
# olora_sketch_fd_divergence campaign (2026-10-21 user design): quantifies
# how SketchLoRA's and FD Sum's fixed-rank sketches diverge from O-LoRA's
# unbounded-rank concatenated adapter over training -- spectral norm of the
# dense-update difference, plus CKA between final features (cumulative, and
# acute pre/post-sketch for SketchLoRA/FD Sum only). 3 methods (olora,
# sketchlora [merge_op=randsvd], fdsum [merge_op=freqdir_dense]) x 3 datasets
# (cifar224-10t, imagenetr-20t, omnibenchmark1k-100t) x 3 seeds = 27 runs.
#
# Each run writes, in addition to the usual metrics JSON:
#   run_logs/olora_sketch_fd_divergence/<method>/<dataset>/snapshots/s<seed>/
#     <olora|sketch>_boundary<n>.pt      -- low-rank (B,A) factors, every task
#   run_logs/olora_sketch_fd_divergence/<method>/<dataset>/features/s<seed>/
#     <olora|sketch>_task<r>_boundary<n>.npz  -- cumulative final features
#     exact_task<n>.npz / sketch_task<n>_boundary<n>.npz (sketchlora/fdsum
#       only) -- acute pre/post-sketch feature pair
# See scripts/analyze_olora_sketch_fd_divergence.py for the offline metric
# computation (spectral-norm-of-difference + CKA) once all three methods for
# a given (dataset, seed) have finished -- NOT run from this script, since it
# needs all three methods' snapshots present and is cheap/fast enough to run
# locally afterward rather than occupying this job's GPU time.
#
# O-LoRA and SketchLoRA/FD-Sum share the identical class_order per
# (dataset, seed) by construction -- all three configs use shuffle=true with
# the SAME seed (no explicit class_order list), and utils/data_manager.py
# seeds np.random identically regardless of model_name. See
# scripts/gen_olora_sketch_fd_divergence_configs.py's own docstring.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every other
# campaign script here (e.g. sketchlora_orth_wave1_datasets.slurm).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <olora|sketchlora_align>/metrics_<tag>.json's own status=="done" -- this
# campaign's own prefix namespaces every tag).
#
# Timing is genuinely unknown for this exact instrumented configuration
# (feature-dump + snapshot overhead on top of normal training) -- 96h wall
# time is a generous guess given Omnibenchmark-1k-100t alone has taken up to
# several hours per run for related campaigns. Relies on the RESUMABLE design
# to make partial progress across resubmissions if one window isn't enough.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=olora_sketch_fd_divergence
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
# same idiom as scripts/ce_final_all.slurm.
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
python scripts/gen_olora_sketch_fd_divergence_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local outfile="${OUT_DIR}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  if [ -f "$outfile" ]; then
    mv "$outfile" "${outfile}.prev-${SLURM_JOB_ID:-unknown}"
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

echo "==== ${CAMPAIGN} start: 3 methods x 3 datasets x 3 seeds, single GPU $(date) ===="

METHODS="olora sketchlora fdsum"
DATASETS="cifar224 imagenetr omnibenchmark1k"
SEEDS="1993 1996 1999"

# Dataset outer loop, not method outer loop: O-LoRA's reference snapshots for
# a given (dataset, seed) are what SketchLoRA/FD-Sum's divergence is measured
# AGAINST, but the analysis script reads all three independently from disk --
# order across methods doesn't matter for correctness, only that all three
# eventually finish for a given (dataset, seed) before
# analyze_olora_sketch_fd_divergence.py is run. Grouped by dataset purely so
# a partial/interrupted run leaves one dataset's full method x seed grid
# mostly together in the logs, easier to audit.
for DATASET in $DATASETS; do
  for METHOD in $METHODS; do
    for SEED in $SEEDS; do
      run_one "$CONFIG_DIR/${METHOD}_${DATASET}_s${SEED}.json"
    done
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
