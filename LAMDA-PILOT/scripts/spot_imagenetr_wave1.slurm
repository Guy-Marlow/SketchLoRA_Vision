#!/bin/bash
#SBATCH --job-name=spot_imagenetr_wave1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_imagenetr_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_imagenetr_wave1/slurm_%j.err
#
# SPOT task-pair similarity preprocessing (2026-09-17), ImageNet-R-20t, 3
# seeds. scripts/spot_task_similarity.py: one-step-probe SPOT (eq3), ResNet34
# from scratch, batch_size=48 (project default -- measured faster AND
# better-converging than 128 at this fixed lr), max_epoch=200, patience=15
# (essentially never fires at this lr -- expect the full 200 epochs every
# time), runs=5 (averaged per seed, matching SPOT's own run_times idea).
#
# Local A5000 measurement extrapolates to ~114h sequential for all 3 seeds;
# H200s estimated ~2x an A5000 -> ~57h, inside this 72h window with margin.
#
# SINGLE GPU BY CONSTRUCTION -- cluster is shared/contested (8 cards total).
#
# RESUMABLE AT THE SEED level only (skips a seed once its output JSON
# exists) -- NOT resumable mid-seed; a job killed partway through a seed's 5
# runs restarts that seed from scratch on resubmission (the python script has
# no internal checkpointing).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=spot_imagenetr_wave1
OUT_DIR="exps/spot_splits"
LOG_DIR="run_logs/$CAMPAIGN"
mkdir -p "$OUT_DIR" "$LOG_DIR"

LOCKFILE="$LOG_DIR/.$CAMPAIGN.lock"
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

PREFLIGHT_ERR="$(mktemp)"
NGPU=$(python -c "import torch, torchvision; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>"$PREFLIGHT_ERR")
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/torchvision import, CUDA unavailable, or non-numeric device count). Details:"
  cat "$PREFLIGHT_ERR"
  rm -f "$PREFLIGHT_ERR"
  exit 1
fi
rm -f "$PREFLIGHT_ERR"
if [ "$NGPU" != "1" ]; then
  echo "NOTE: torch.cuda.device_count()=${NGPU} (expected 1 for this job's --gres=gpu:1) -- proceeding on CUDA_VISIBLE_DEVICES=0 regardless."
fi
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

SEEDS="1993 1996 1999"

for SEED in $SEEDS; do
  OUT="$OUT_DIR/imagenetr_s${SEED}.json"
  if [ -f "$OUT" ]; then
    echo "==== SKIP (done) seed ${SEED} $(date) ===="
    continue
  fi
  LOGFILE="${LOG_DIR}/imagenetr_s${SEED}.out"
  if [ -f "$LOGFILE" ]; then
    mv "$LOGFILE" "${LOGFILE}.prev-${SLURM_JOB_ID:-unknown}"
  fi
  echo "==== START seed ${SEED} $(date) ===="
  python scripts/spot_task_similarity.py \
    --dataset imagenetr --seed "$SEED" --runs 5 \
    --max_epoch 200 --patience 15 --backbone resnet34 \
    --out "$OUT" \
    > "$LOGFILE" 2>&1 || true
  echo "==== DONE  seed ${SEED} $(date) ===="
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
