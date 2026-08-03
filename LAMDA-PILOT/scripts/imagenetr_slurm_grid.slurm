#!/bin/bash
#SBATCH --job-name=imagenetr_slurm_grid
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/imagenetr_slurm_grid/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/imagenetr_slurm_grid/slurm_%j.err
#
# ImageNet-R counterpart to scripts/round2_slurm_grid.slurm (user request
# 2026-07-31): same 45-cell shape (SeqLoRA, SketchLoRA, O-LoRA, InfLoRA,
# TreeLoRA x {50, 100, 200}MB x 3 seeds {1993, 1996, 1999}), bounded_memory
# boundary mode, but on ImageNet-R's full 20-task/200-class split instead of
# OmniBenchmark-1K -- structurally identical script, mechanically copied from
# round2_slurm_grid.slurm (same lockfile/preflight/FIFO-worker-queue idiom),
# pointed at a NEW config dir (exps/imagenetr_slurm_grid/) and a NEW,
# dedicated run_logs subfolder (run_logs/imagenetr_slurm_grid/) so this never
# reads, writes, or collides with round2_slurm_grid's own data, the
# completion-and-50mb script's data, or the existing separate (2-seed,
# locally-run) exps/imagenetr_grid/ campaign.
#
# This is a FRESH campaign -- nothing in exps/imagenetr_slurm_grid/ or
# run_logs/imagenetr_slurm_grid/ exists yet, so (unlike the completion+50mb
# script) the job list here is the full 45-cell cross product, not a
# hand-picked subset; there is nothing pre-existing it could overwrite.
#
# Wall time (40h) is an estimate, not a measurement: ImageNet-R has ~24,076
# training images vs OmniBenchmark-1K's ~168,718 (~1/7th), so bounded_memory
# cycle counts and per-run wall-clock should be well under round2_slurm_grid's
# observed 2.5-3.5h/run at 100-200MB -- consistent with the existing 2-seed
# local imagenetr_grid campaign's own queue-log timings (~1-1.3h/run at these
# budgets) -- but 50MB is a new, smaller, more-cycles budget point that
# campaign never tested, so this is a margin-padded estimate, not a
# validated number.
#
# RESUMABLE / LOCKFILE: identical convention to round2_slurm_grid.slurm --
# skips any run whose accuracy json has "partial": false AND whose metrics
# json has "status": "done"; refuses to start a second instance of itself.
# Does NOT check round2_slurm_grid's or the completion script's lockfiles --
# this grid uses fresh directories and a fresh SLURM job, so it can safely
# run concurrently with those if the cluster has GPUs free elsewhere; it only
# guards against colliding with a second copy of ITSELF.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/imagenetr_slurm_grid

LOCKFILE="run_logs/imagenetr_slurm_grid/.imagenetr_slurm_grid.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another imagenetr_slurm_grid.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check (same convention as round2_slurm_grid.slurm).
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/imagenetr_slurm_grid_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/imagenetr_slurm_grid_preflight_$$.err
  rm -f /tmp/imagenetr_slurm_grid_preflight_$$.err
  exit 1
fi
rm -f /tmp/imagenetr_slurm_grid_preflight_$$.err

# utils/data.py:239-240 hardcodes ./data/imagenet-r/{train,test} (hyphenated,
# matching data_prep.py's TARGET_DIR convention), but on this cluster the data
# is materialized at ./data/imagenetr (no hyphen) -- a naming mismatch, not
# missing data. Symlink rather than re-download or rename: idempotent, fully
# reversible, doesn't touch the existing directory.
if [ ! -e ./data/imagenet-r ] && [ -d ./data/imagenetr ]; then
  ln -s imagenetr ./data/imagenet-r
  echo "==== symlinked ./data/imagenet-r -> ./data/imagenetr $(date) ===="
fi

echo "==== config generation start $(date) ===="
python scripts/gen_imagenetr_slurm_grid_configs.py
echo "==== config generation done $(date) ===="

BUDGETS=(50 100 200)
METHODS=(seqlora sketchlora olora inflora treelora)
SEEDS=(1993 1996 1999)
NGPU=2

echo "==== imagenetr_slurm_grid start: ${#BUDGETS[@]} budgets x ${#METHODS[@]} methods x ${#SEEDS[@]} seeds, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=budget $3=method $4=seed
  local gpu="$1" budget="$2" method="$3" seed="$4"
  local tag="${method}_${budget}mb_s${seed}"
  local config="exps/imagenetr_slurm_grid/${tag}.json"
  local prefix="imagenetr_slurm_grid_${budget}mb_${method}_s${seed}"
  # matches trainer.py::_run_bounded_memory's exact output path
  local json="run_logs/boundedmem_${method}_${prefix}_s${seed}.json"
  # matches models/bounded_memory_mixin.py's MetricsLogger tag/out_path exactly
  local metrics_json="run_logs/final/${method}/metrics_${method}_${prefix}_s${seed}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$json" ] && grep -q '"partial": false' "$json" \
     && [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done, accuracy+metrics both complete) ${tag} $(date) ===="
    return
  fi
  echo "==== START [gpu ${gpu}] ${tag} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --config "$config" \
    > "run_logs/imagenetr_slurm_grid/${tag}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${tag} $(date) ===="
}

for BUDGET in "${BUDGETS[@]}"; do
  echo "==== BUDGET ${BUDGET}MB start $(date) ===="

  JOBS_FIFO="run_logs/imagenetr_slurm_grid/.jobs_${BUDGET}mb.fifo"
  rm -f "$JOBS_FIFO"
  mkfifo "$JOBS_FIFO"

  (
    for SEED in "${SEEDS[@]}"; do
      for METHOD in "${METHODS[@]}"; do
        echo "${METHOD}|${SEED}"
      done
    done
  ) > "$JOBS_FIFO" &
  FIFO_WRITER_PID=$!

  worker() {
    local gpu="$1"
    local method seed
    while IFS='|' read -r method seed; do
      run_one "$gpu" "$BUDGET" "$method" "$seed"
    done < "$JOBS_FIFO"
  }

  for (( g=0; g<NGPU; g++ )); do
    worker "$g" &
  done
  wait
  wait "$FIFO_WRITER_PID" 2>/dev/null || true
  rm -f "$JOBS_FIFO"

  echo "==== BUDGET ${BUDGET}MB done (all methods, all seeds) $(date) ===="
done
echo "==== ALL RUNS COMPLETE (or wall reached) $(date) ===="
