#!/bin/bash
#SBATCH --job-name=final_vision
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/final_vision/slurm_%j.err
#
# Final vision CL grid: 10 methods x 4 datasets x 3 seeds = 120 runs, full
# (non-truncated) 20-task splits, batch_size=48/lr=3e-4 shared HP convention
# (ProgPrompt overridden to 15 epochs -- see scripts/gen_final_vision_configs.py).
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
# "status": "done", so re-submitting the same script after a 144h wall-kill
# picks up exactly where it left off. Each run writes its JSON incrementally
# (crash-safe) and `|| true` after the python call means one crash/OOM can't
# abort the rest of the sweep.
#
# >>> PRE-SUBMIT (run once, on a node with internet access) <<<
#   bash scripts/setup_cluster.sh                    # datasets + timm PTM weights
#   python scripts/gen_final_vision_configs.py        # writes exps/final_vision/*.json
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION"
mkdir -p run_logs/final_vision

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

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
