#!/bin/bash
#SBATCH --job-name=sketchlora_orth_wave1_datasets
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_wave1_datasets/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_wave1_datasets/slurm_%j.err
#
# Full (orth) SketchLoRA across the wave1 dataset splits, minus
# OmniBenchmark-1k (2026-09-04 user request -- fills in the master table's
# remaining SketchLoRA cells). 4 datasets x 3 seeds = 12 runs:
#   CIFAR-100/cifar224-10t, ImageNet-R-20t, Food101-20t, SUN397-10t.
# svd_rank=10, sketchlora_align_weight=0.5 (the standing "SketchLoRA"
# default). See scripts/gen_sketchlora_orth_wave1_datasets_configs.py's own
# docstring for the exact per-dataset split params (copied verbatim from
# exps/wave1_final/olora_<dataset>_s1993.json, same as every other method in
# the master table) and the SUN397 task-count verification (init_cls=37,
# increment=40 -- confirmed 37 + 40*9 = 397 exactly, already a clean 10-task
# split, nothing needed fixing).
#
# SINGLE GPU BY CONSTRUCTION (this cluster only has one) -- same plain
# sequential pattern as scripts/ce_final_all.slurm / ce_final_all_retry3.slurm
# / the corrected scripts/sketchlora_orth_sens_ablations.slurm: no FIFO, no
# worker-pool, CUDA_VISIBLE_DEVICES pinned to 0. --gres=gpu:1/8 cpu/48G,
# same resource request as every other single-GPU SketchLoRA campaign here.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- this campaign's
# own prefix namespaces every tag, so it can only ever match a genuine
# attempt from THIS campaign).
#
# NOTE: this reruns ImageNet-R-20t fresh even though a 3-seed (svd_rank=10,
# align_weight=0.5) orth result already exists locally under a different
# prefix (exps/sketchlora_fixedrank_orth05_imagenetr_s*.json) -- included
# here anyway so all 4 master-table cells come from one uniform campaign.
#
# 48h wall time for 12 runs on one GPU: ImageNet-R/CIFAR-100-scale SketchLoRA
# runs have taken ~2.5-5h each elsewhere in this project; Food101 (20 tasks)
# and SUN397 (10 tasks, but very wide first task at 37 classes) are untested
# at this rank/config so their per-run time isn't yet known. Relies on the
# RESUMABLE design above to make partial progress across resubmissions if
# one 48h window isn't enough for all 12 -- not a bug, expected.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_orth_wave1_datasets
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
python scripts/gen_sketchlora_orth_wave1_datasets_configs.py
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

echo "==== ${CAMPAIGN} start: 4 datasets x 3 seeds, single GPU $(date) ===="

DATASETS="cifar224 imagenetr food101 sun397"
SEEDS="1993 1996 1999"

for DATASET in $DATASETS; do
  for SEED in $SEEDS; do
    run_one "$CONFIG_DIR/${DATASET}_s${SEED}.json"
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
