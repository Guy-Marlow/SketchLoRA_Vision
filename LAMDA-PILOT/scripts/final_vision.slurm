#!/bin/bash
#SBATCH --job-name=final_vision
#SBATCH --time=180:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.err
#
# Final vision CL grid: 10 methods x 4 datasets x 3 seeds = 120 runs, full
# (non-truncated) 20-task splits, batch_size=48/lr=3e-4 shared HP convention
# (ProgPrompt overridden to 15 epochs -- see scripts/gen_final_vision_configs.py).
# --cpus-per-task=16 / --mem=128G = 2x an individual run's own footprint
# (num_workers=8, ~64G), since two runs now train concurrently (see below).
#
# TWO GPUs, ONE job (still not a SLURM array): request both GPUs once in the
# header, then run two background `python` workers per method pinned to GPU 0
# and GPU 1 via CUDA_VISIBLE_DEVICES -- mirrors the worker-pool pattern already
# used by the language-track grids (tokmem/atomic/run_experiments.slurm,
# run_longhorizon_750_v2.slurm), just with the pool re-formed fresh per method
# rather than over the whole run (see METHOD-MAJOR below for why).
#
# METHOD-MAJOR ordering (user-specified 2026-07-19, preserved under the 2-GPU
# parallelization added 2026-07-20): each method still runs to full completion
# -- all 4 datasets x 3 seeds (12 runs) -- before the next method starts. What
# changed: within a single method's 12 runs, both GPUs now work through them
# in parallel (round-robin split), and the script waits for BOTH lanes to
# finish that method before advancing -- roughly halves wall-clock time
# without breaking the "each method completes before the next starts"
# guarantee. Order: SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA, CL-LoRA,
# RainbowPrompt, EASE, TUNA, ProgPrompt.
#
# RESUMABLE: skips any run whose final_metrics JSON already says
# "status": "done", so re-submitting the same script after a 180h wall-kill
# picks up exactly where it left off. Each run writes its JSON incrementally
# (crash-safe) and `|| true` after each python call means one crash/OOM can't
# abort the rest of the sweep (or its GPU-lane partner).
#
# LOCKFILE: refuses to start a second instance while one is already running
# (e.g. an accidental double-submission) -- two instances racing on the same
# config would double-launch a run and corrupt its metrics JSON/output file.
#
# SELF-CONTAINED: dataset prep (scripts/setup_cluster.sh, which also prefetches
# the 2 timm PTM checkpoints) and config generation both run automatically at
# the start of THIS job, below -- no manual pre-submit step needed, just
# `sbatch` this file. Safe to do unconditionally because this still runs as a
# single up-front step before either GPU lane starts (no concurrent-download
# race). Both scripts are already idempotent (skip anything already present),
# so re-submitting this script after a wall-kill doesn't redo finished
# downloads -- CIFAR-100 in particular can be SLOW (cs.toronto.edu is a
# single-connection, unaccelerated academic host, ~40min/169MB observed) --
# expect this step to take a while on first run, near-instant on resubmission.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/final_vision

LOCKFILE="run_logs/final_vision/.final_vision.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another final_vision.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check: fail fast and loudly if conda activation didn't
# actually put the right environment on PATH (e.g. wrong CONDA_ENV_NAME, or a
# path assumption above that doesn't hold on this cluster) -- without this,
# a broken env would silently burn the whole wall-clock budget: `python
# main.py` would ImportError on every single one of the 120 runs, each
# swallowed by `|| true` below, and the job would report "ALL RUNS COMPLETE"
# at the end having done nothing.
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/final_vision_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/final_vision_preflight_$$.err
  rm -f /tmp/final_vision_preflight_$$.err
  exit 1
fi
rm -f /tmp/final_vision_preflight_$$.err

echo "==== data prep + config generation start $(date) ===="
bash scripts/setup_cluster.sh
python scripts/gen_final_vision_configs.py
echo "==== data prep + config generation done $(date) ===="

METHODS=(seqlora sketchlora olora inflora treelora cllora rainbowprompt ease tuna progprompt)
DATASETS=(cifar224 imagenetr food101 sun397)
SEEDS=(1993 1994 1995)
NGPU=2

echo "==== final_vision grid start: ${#METHODS[@]} methods x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=method $3=dataset $4=seed
  local gpu="$1" method="$2" dataset="$3" seed="$4"
  local tag="${dataset}_${method}_s${seed}"
  local config="exps/final_vision/${tag}.json"
  # Mirrors trainer.py's exact tag format: "{model_name}_{dataset}_{prefix}_s{seed}",
  # where gen_final_vision_configs.py sets prefix="final_vision_${tag}" -- so the
  # dataset/method/seed appear twice in the metrics filename. Not a typo; matches
  # what trainer.py actually writes.
  local json="run_logs/final/${method}/metrics_${method}_${dataset}_final_vision_${tag}_s${seed}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$json" ] && grep -q '"status": "done"' "$json"; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START [gpu ${gpu}] ${tag} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --config "$config" \
    > "run_logs/final_vision/${tag}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${tag} $(date) ===="
}

for METHOD in "${METHODS[@]}"; do
  echo "==== METHOD ${METHOD} start $(date) ===="

  RUNS=()
  for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUNS+=("${DATASET}|${SEED}")
    done
  done

  worker() {
    local gpu="$1"
    local i ds seed
    for (( i=gpu; i<${#RUNS[@]}; i+=NGPU )); do
      IFS='|' read -r ds seed <<< "${RUNS[$i]}"
      run_one "$gpu" "$METHOD" "$ds" "$seed"
    done
  }

  for (( g=0; g<NGPU; g++ )); do
    worker "$g" &
  done
  wait

  echo "==== METHOD ${METHOD} done (both GPU lanes finished) $(date) ===="
done
echo "==== ALL RUNS COMPLETE (or wall reached) $(date) ===="
