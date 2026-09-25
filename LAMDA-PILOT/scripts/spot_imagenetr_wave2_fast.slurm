#!/bin/bash
#SBATCH --job-name=spot_imagenetr_wave2_fast
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_imagenetr_wave2_fast/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_imagenetr_wave2_fast/slurm_%j.err
#
# SPOT task-pair similarity preprocessing, faster/cheaper variant of
# spot_imagenetr_wave1.slurm (2026-09-17), ImageNet-R-20t THEN CIFAR-100
# (cifar224) after it, 3 seeds each. scripts/spot_task_similarity.py:
# one-step-probe SPOT (eq3), ResNet34 from scratch, batch_size=48 (project
# default). CHANGED from wave1: max_epoch 200->100, patience 15->10, runs
# 5->3, and CIFAR-100 added as a second dataset (wave1 was ImageNet-R only).
# wave1's own comment noted patience=15 essentially never fired at its lr --
# that claim is NOT re-asserted here for patience=10; whether it fires more
# often at this lower threshold is unverified, expect actual per-task epoch
# counts to vary run to run.
#
# Writes to a SEPARATE output location (exps/spot_splits_fast/, distinct
# campaign name/log dir) rather than exps/spot_splits/ -- deliberately does
# NOT touch or overwrite wave1's original 200-epoch/patience-15/runs-5
# ImageNet-R-only splits, which are already in use for the similar/dissimilar
# task-ordering experiments elsewhere in this project.
#
# Rough time estimate (ImageNet-R phase only, CIFAR-100's own per-run cost
# not separately measured): half the epoch budget and 3/5 the run count vs
# wave1 (~114h sequential for all 3 ImageNet-R seeds on a local A5000)
# suggests roughly 100/200 * 3/5 =~ 0.3x wave1's ImageNet-R wall time -> very
# roughly ~34h on an A5000 equivalent for the ImageNet-R phase, ~17h at H200's
# ~2x speed. CIFAR-100 adds unknown additional wall time on top of that (not
# estimated here) -- a rough scaling guess either way, not measured for this
# specific config. --time bumped to 60h (up from the 36h sized for the
# ImageNet-R-only estimate alone) to leave real margin for CIFAR-100's
# unmeasured additional cost; still a guess, not a measured bound.
#
# SINGLE GPU BY CONSTRUCTION -- cluster is shared/contested (8 cards total).
#
# RESUMABLE AT THE SEED level only (skips a seed once its output JSON
# exists) -- NOT resumable mid-seed; a job killed partway through a seed's 3
# runs restarts that seed from scratch on resubmission (the python script has
# no internal checkpointing).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=spot_imagenetr_wave2_fast
OUT_DIR="exps/spot_splits_fast"
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
# Lineup: ImageNet-R fully (all 3 seeds) first, then CIFAR-100 (cifar224)
# after it -- despite the job/campaign name (kept as-is from the
# ImageNet-R-only original), this now covers both datasets.
DATASETS="imagenetr cifar224"

for DATASET in $DATASETS; do
  for SEED in $SEEDS; do
    OUT="$OUT_DIR/${DATASET}_s${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "==== SKIP (done) ${DATASET} seed ${SEED} $(date) ===="
      continue
    fi
    LOGFILE="${LOG_DIR}/${DATASET}_s${SEED}.out"
    if [ -f "$LOGFILE" ]; then
      mv "$LOGFILE" "${LOGFILE}.prev-${SLURM_JOB_ID:-unknown}"
    fi
    echo "==== START ${DATASET} seed ${SEED} $(date) ===="
    python scripts/spot_task_similarity.py \
      --dataset "$DATASET" --seed "$SEED" --runs 3 \
      --max_epoch 100 --patience 10 --backbone resnet34 \
      --out "$OUT" \
      > "$LOGFILE" 2>&1 || true
    echo "==== DONE  ${DATASET} seed ${SEED} $(date) ===="
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
