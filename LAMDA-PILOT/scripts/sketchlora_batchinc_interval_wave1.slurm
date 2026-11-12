#!/bin/bash
#SBATCH --job-name=sketchlora_batchinc_interval_wave1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_batchinc_interval_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_batchinc_interval_wave1/slurm_%j.err
#
# SketchLoRA batch-incremental + KL-drift-detection sketch-interval sweep
# (2026-11-12 user request): CIFAR-100-10t and ImageNet-R-20t, FULL task
# count (no stop_after_tasks -- a real analysis campaign, not a truncated
# smoke test), 5 ABSOLUTE sketch-interval-step values per dataset x 3
# seeds (1993/1996/1999) = 30 runs. boundary_mode=="kl_batch_inc"
# throughout -- the REAL KL drift detector governs adapter spawning in
# every run, independent of the sketch-interval clock being swept here.
#
# STEP INCREMENTS (not a ratio of task 0's own step count -- an earlier
# version of this campaign used that design, replaced on explicit
# instruction): imagenetr=200/400/600/800/1000 (task 0 itself is 540
# steps, mean ~509.7 steps/task across all 20 tasks x 3 seeds);
# cifar224=1000/1500/2000/2500/3000 (every task is exactly 2100 steps --
# CIFAR-100 is a fixed 500 images/class x 10 classes/task).
#
# HYPERPARAMETERS (full detail in scripts/gen_sketchlora_batchinc_
# interval_wave1_configs.py's own docstring): settled SketchLoRA
# convention (init_lr=0.001 -- SAME for both datasets, batch_size=48,
# tuned_epoch=20, lora_rank=10, svd_rank=10, batchinc_orth_weight=0.5)
# plus this session's own validated detector/fold-clock defaults
# (batchinc_pending_cap=3, batchinc_kl_window_samples=256,
# batchinc_kl_tau=4.0, batchinc_kl_consecutive=3,
# batchinc_kl_warmup_multiplier=2, batchinc_lr_warmup_steps=20).
#
# Config generator: scripts/gen_sketchlora_batchinc_interval_wave1_
# configs.py (run at the top of this script, same as every other
# campaign here).
#
# OUTPUT / RESUMABILITY: boundary_mode=="kl_batch_inc" does NOT use the
# run_logs/final/<model>/metrics_*.json / final_metrics convention at all
# -- it writes run_logs/kl_batch_inc_<model_name>_<prefix>_s<seed>.json via
# trainer.py::_run_stream. scripts/_bankcap_run_done.py (used by every
# other campaign script here) would therefore ALWAYS report "not done" for
# these runs and resubmit them from scratch even after a clean finish --
# this script instead uses the matching scripts/_kl_batch_inc_run_done.py
# check. Each config's `prefix` is unique per (dataset, interval, seed),
# so every run's output path is distinct.
#
# NO MID-RUN CHECKPOINTING EXISTS for this setting (models/
# sketchlora_batchinc.py::kl_batch_inc_run has no incremental save) -- the
# output file above is written exactly ONCE, after a run's full task loop
# completes. A run still in progress when the wall-time limit hits is
# killed with NO partial credit and restarts from task 0 on the next
# submission, unlike every "final_metrics"-style campaign in this project.
#
# RUNTIME: NOT estimated here. This session's only per-step timing data
# for this setting comes from the LOCAL single-A5000 machine, which is
# different hardware from this cluster's GPU -- extrapolating a cluster
# wall-time estimate from that number would be invalid, not merely
# imprecise, so none is given. Watch the first few completed runs' actual
# elapsed time on this cluster before assuming how far one 96h submission
# will get.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every
# other campaign script in this project (this cluster has exactly one GPU
# available).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's
# log -- run_one() below backs up any existing outfile first (same
# convention as every other campaign script here).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_batchinc_interval_wave1
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
python scripts/gen_sketchlora_batchinc_interval_wave1_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. cifar224/interval2000
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_kl_batch_inc_run_done.py "$config" 2>/dev/null; then
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

# Run order: cifar224 then imagenetr (the order the user listed them in),
# interval ascending within each, seed ascending within each interval --
# `sort` keeps this deterministic from the glob below.
echo "==== ${CAMPAIGN} start: 30 runs, single GPU $(date) ===="

for dataset in cifar224 imagenetr; do
  for interval in $(python -c "
import sys; sys.path.insert(0, 'scripts')
from gen_sketchlora_batchinc_interval_wave1_configs import DATASETS
print(' '.join(str(x) for x in DATASETS['${dataset}']['intervals']))
"); do
    while IFS= read -r config; do
      run_one "$config"
    done < <(find "$CONFIG_DIR/$dataset/interval${interval}" -name "*.json" 2>/dev/null | sort)
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
