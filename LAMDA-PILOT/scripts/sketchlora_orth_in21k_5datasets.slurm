#!/bin/bash
#SBATCH --job-name=sketchlora_orth_in21k_5datasets
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_in21k_5datasets/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_in21k_5datasets/slurm_%j.err
#
# Full (orth) SketchLoRA across all 5 vision benchmarks, ImageNet-21k-ONLY
# pretrained backbone (2026-09-07 user request) -- 15 runs (5 datasets x 3
# seeds). svd_rank=10, sketchlora_align_weight=0.5 (the standing "SketchLoRA"
# default). backbone_type=vit_base_patch16_224_in21k_lora resolves via timm
# to Google's AugReg checkpoint WITHOUT the ImageNet-1k fine-tuning stage
# (confirmed directly against timm's registered default_cfgs before writing
# this -- num_classes=21843, no "--imagenet2012-..." suffix on the checkpoint
# URL, unlike this project's usual vit_base_patch16_224_lora backbone). See
# scripts/gen_sketchlora_orth_in21k_5datasets_configs.py's own docstring for
# the exact per-dataset split params and the timm verification.
#
# UNLIKE sketchlora_orth_wave1_datasets, OmniBenchmark-1k IS included here --
# this is a new backbone-pretraining axis, not duplicate work; no existing
# OmniBenchmark-1k result uses this backbone.
#
# WALL TIME (96h, up from the 4-dataset campaign's 48h): OmniBenchmark-1k is
# a 100-task split -- at this project's typical ~1-1.5 min/task pace for
# SketchLoRA, that alone is several hours PER SEED, x3 seeds, on top of the
# other 4 (smaller) datasets x3 seeds each. Almost certainly will NOT finish
# in one submission; relies on the RESUMABLE design below (same as every
# other campaign here) to make partial progress across resubmissions.
#
# SINGLE GPU BY CONSTRUCTION (this cluster only has one) -- same plain
# sequential pattern as every other single-GPU campaign here.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- this
# campaign's own prefix namespaces every tag).
#
# ORDER: dataset-outer, seed-inner (ALL 3 seeds of one dataset before moving
# to the next), OmniBenchmark-1k LAST -- so a resubmission that runs out of
# wall time still banks COMPLETE 3-seed mean+-std for as many datasets as
# possible, rather than 1-seed partial coverage of all 5 (seed-outer would
# risk never finishing even a single dataset's full 3 seeds if interrupted).
# Smallest/fastest datasets (cifar224, sun397 -- both 10-task) first purely
# so something usable banks soonest if the job is cut short early.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_orth_in21k_5datasets
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
python scripts/gen_sketchlora_orth_in21k_5datasets_configs.py
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

echo "==== ${CAMPAIGN} start: 5 datasets x 3 seeds, single GPU $(date) ===="

# dataset-outer, seed-inner, smallest/fastest first -- see header note.
DATASETS="cifar224 sun397 imagenetr food101 omnibenchmark1k"
SEEDS="1993 1996 1999"

for DATASET in $DATASETS; do
  for SEED in $SEEDS; do
    run_one "$CONFIG_DIR/${DATASET}_s${SEED}.json"
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
