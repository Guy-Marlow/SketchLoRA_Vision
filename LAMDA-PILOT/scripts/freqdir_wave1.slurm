#!/bin/bash
#SBATCH --job-name=freqdir_wave1
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_wave1/slurm_%j.err
#
# freqdir merge_op campaign, wave 1 (2026-09-15 user request): decoupled-space
# Frequent-Directions SketchLoRA -- never forms the dense delta_W; the
# sketch's input (A) and output (B) directions are extracted from two
# UNRELATED concatenate-then-shrink-then-truncate decompositions instead of
# one joint SVD of B@A. See utils/freqdir.py's module docstring for the full
# algorithm and models/sketchlora.py's merge_op="freqdir" branch for the
# wiring. 2 benchmarks x 3 seeds = 6 runs.
#
# Each config is exps/sketchlora_orth_wave1_datasets/<dataset>_s<seed>.json
# (the existing baseline SketchLoRA config already used for the master-table
# numbers) loaded VERBATIM and only merge_op/sketch_diag/prefix overridden --
# see scripts/gen_freqdir_wave1_configs.py's own docstring. sketch_diag is
# explicitly off (not auto-disabled-with-warning at runtime) since freqdir
# never builds what that diagnostic reads.
#
# SINGLE GPU BY CONSTRUCTION (this cluster only has one) -- same plain
# sequential pattern as every other single-GPU campaign here.
#
# WALL TIME (72h): CIFAR-100 (10 tasks) and ImageNet-R (20 tasks) at this
# project's standard 20-epoch budget took roughly 45min-1.5h per run for the
# randsvd baseline earlier this session -- freqdir's own per-merge cost is
# LOWER (two thin (2*svd_rank)-wide exact SVDs vs. randsvd's randomized
# projection, and no dense delta_W ever formed), so 6 runs should comfortably
# fit inside 72h with room to spare; the longer window is headroom, not an
# expectation of needing it.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- this
# campaign's own prefix namespaces every tag, so it can only ever match a
# genuine attempt from THIS campaign).
#
# ORDER: dataset-outer, seed-inner (all 3 seeds of one dataset before moving
# to the next), smallest/fastest (cifar224, 10 tasks) first -- so a
# resubmission that runs out of time still banks a complete 3-seed dataset
# rather than partial coverage of both.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=freqdir_wave1
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
python scripts/gen_freqdir_wave1_configs.py
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

echo "==== ${CAMPAIGN} start: 2 datasets x 3 seeds, single GPU $(date) ===="

# dataset-outer, seed-inner, smallest/fastest first -- see header note.
DATASETS="cifar224 imagenetr"
SEEDS="1993 1996 1999"

for DATASET in $DATASETS; do
  for SEED in $SEEDS; do
    run_one "$CONFIG_DIR/${DATASET}_s${SEED}.json"
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
