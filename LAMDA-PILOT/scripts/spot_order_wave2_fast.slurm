#!/bin/bash
#SBATCH --job-name=spot_order_wave2_fast
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_order_wave2_fast/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/spot_order_wave2_fast/slurm_%j.err
#
# SketchLoRA (orth) on similar-adjacent vs dissimilar-adjacent task orderings
# derived from the "wave2_fast" SPOT similarity data (100 epochs, patience
# 10, 3 runs -- see scripts/spot_imagenetr_wave2_fast.slurm), on both
# ImageNet-R-20t and CIFAR-100/cifar224-10t, 3 seeds each:
#   2 datasets x 2 orderings x 3 seeds = 12 runs.
# Configs generated at runtime (see "config generation" step below) by
# scripts/gen_spot_order_wave2_fast_configs.py from
# exps/spot_splits_fast/<dataset>_s<seed>.json (must already exist --
# committed to the repo, produced by spot_imagenetr_wave2_fast.slurm and
# pulled locally) plus the baseline
# exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json (svd_rank=10,
# sketchlora_align_weight=0.5, the standing SketchLoRA default) -- only
# shuffle/class_order/prefix are overridden, everything else is the baseline
# config verbatim. Each ordering keeps every task's own class set; only the
# task SEQUENCE changes (greedy nearest-neighbor chain over the SPOT
# similarity matrix, maximizing or minimizing adjacent-task similarity).
#
# Distinct campaign/prefix from the earlier spot_order_wave1 (which used the
# original 200-epoch/patience-15/runs-5 SPOT data, ImageNet-R only) -- this
# is a fresh, separate set of runs, not a resubmission of that one.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as
# scripts/sketchlora_orth_wave1_datasets.slurm.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- this
# campaign's own prefix namespaces every tag, so it can only ever match a
# genuine attempt from THIS campaign).
#
# Wall time not measured for this exact config; CIFAR-100/ImageNet-R-scale
# SketchLoRA runs have taken ~2.5-5h each elsewhere in this project, so 12
# runs could exceed one 48h window -- the RESUMABLE design handles that
# across resubmissions, not a bug if it doesn't finish in one pass.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=spot_order_wave2_fast
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
# same idiom as scripts/sketchlora_orth_wave1_datasets.slurm.
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

DATASETS="imagenetr cifar224"
SEEDS="1993 1996 1999"

echo "==== config generation start $(date) ===="
for DATASET in $DATASETS; do
  for SEED in $SEEDS; do
    SPOT_SRC="exps/spot_splits_fast/${DATASET}_s${SEED}.json"
    if [ ! -f "$SPOT_SRC" ]; then
      echo "FATAL: missing SPOT source ${SPOT_SRC} -- expected it committed to the repo before this job runs."
      exit 1
    fi
    python scripts/gen_spot_order_wave2_fast_configs.py --dataset "$DATASET" --seed "$SEED"
  done
done
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

echo "==== ${CAMPAIGN} start: 2 datasets x 2 orderings x 3 seeds, single GPU $(date) ===="

ORDERINGS="similar dissimilar"

for DATASET in $DATASETS; do
  for ORDERING in $ORDERINGS; do
    for SEED in $SEEDS; do
      run_one "$CONFIG_DIR/${DATASET}_${ORDERING}_s${SEED}.json"
    done
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
