#!/bin/bash
#SBATCH --job-name=final_vision
#SBATCH --time=180:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.err
#
# Final vision CL grid: 10 methods x 4 datasets x 3 seeds = 120 runs, full
# (non-truncated) 20-task splits, batch_size=48/lr=3e-4 shared HP convention
# (ProgPrompt overridden to 15 epochs -- see scripts/gen_final_vision_configs.py).
# --cpus-per-task=8 matches every DataLoader's num_workers=8 (models/*.py) --
# dropped by mistake in an earlier rewrite of this script from array-based to
# single-job; restored 2026-07-20 audit.
#
# ONE GPU, ONE job (not a SLURM array): mirrors the single-job style already
# used for the language-track grids (tokmem/atomic/run_experiments.slurm,
# run_longhorizon_750_v2.slurm) -- request the GPU once in the header, then
# run plain sequential `python` calls, no srun/CUDA_VISIBLE_DEVICES juggling.
#
# METHOD-MAJOR ordering (user-specified 2026-07-19): each method runs to
# completion -- all 4 datasets x 3 seeds (12 runs) -- before the next method
# starts. Order: SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA, CL-LoRA,
# RainbowPrompt, EASE, TUNA, ProgPrompt.
#
# RESUMABLE: skips any run whose final_metrics JSON already says
# "status": "done", so re-submitting the same script after a 180h wall-kill
# picks up exactly where it left off. Each run writes its JSON incrementally
# (crash-safe) and `|| true` after the python call means one crash/OOM can't
# abort the rest of the sweep.
#
# LOCKFILE: refuses to start a second instance while one is already running
# (e.g. an accidental double-submission) -- two instances racing on the same
# config would double-launch a run and corrupt its metrics JSON/output file.
#
# >>> PRE-SUBMIT (run once, on a node with internet access) <<<
#   bash scripts/setup_cluster.sh                    # datasets + timm PTM weights
#   python scripts/gen_final_vision_configs.py        # writes exps/final_vision/*.json
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

METHODS=(seqlora sketchlora olora inflora treelora cllora rainbowprompt ease tuna progprompt)
DATASETS=(cifar224 imagenetr food101 sun397)
SEEDS=(1993 1994 1995)

echo "==== final_vision grid start: ${#METHODS[@]} methods x ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds $(date) ===="

for METHOD in "${METHODS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      TAG="${DATASET}_${METHOD}_s${SEED}"
      CONFIG="exps/final_vision/${TAG}.json"
      # Mirrors trainer.py's exact tag format: "{model_name}_{dataset}_{prefix}_s{seed}",
      # where gen_final_vision_configs.py sets prefix="final_vision_${TAG}" -- so the
      # dataset/method/seed appear twice in the metrics filename. Not a typo; matches
      # what trainer.py actually writes.
      JSON="run_logs/final/${METHOD}/metrics_${METHOD}_${DATASET}_final_vision_${TAG}_s${SEED}.json"

      if [ ! -f "$CONFIG" ]; then
        echo "==== SKIP (no config) ${TAG} $(date) ===="
        continue
      fi
      if [ -f "$JSON" ] && grep -q '"status": "done"' "$JSON"; then
        echo "==== SKIP (done) ${TAG} $(date) ===="
        continue
      fi
      echo "==== START ${TAG} $(date) ===="
      python main.py --config "$CONFIG" > "run_logs/final_vision/${TAG}.out" 2>&1 || true
      echo "==== DONE  ${TAG} $(date) ===="
    done
  done
done
echo "==== ALL RUNS COMPLETE (or wall reached) $(date) ===="
