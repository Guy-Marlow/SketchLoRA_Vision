#!/bin/bash
#SBATCH --job-name=final_vision
#SBATCH --array=0-107
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=run_logs/final_vision/slurm_%A_%a.out
#
# >>> CLUSTER-SPECIFIC PLACEHOLDERS -- fill these in before submitting <<<
# #SBATCH --partition=REPLACE_ME       # e.g. h200, gpu, etc.
# #SBATCH --account=REPLACE_ME         # if your cluster requires an allocation/account
#
# Est. 30-45 min/run per Experiments_Timeline.pdf -- 1h wall-clock per array
# task leaves headroom; raise --time if a dataset/method combo runs long.
#
# Usage:
#   sbatch scripts/final_vision.slurm
# One array task = one (dataset, method, seed) config, 108 total
# (9 methods x 4 datasets x 3 seeds). Each task gets its own single-GPU
# allocation; SLURM sets CUDA_VISIBLE_DEVICES so it always appears as device 0
# from inside the job (matches configs' "device": ["0"] from gen_final_vision_
# configs.py -- do not change that field for this launch path).
#
# Dataset setup + config generation + PTM checkpoint prefetch are NOT run
# automatically by this script (a SLURM array launches 108 independent tasks
# in parallel -- doing idempotent setup 108 times concurrently invites a race
# on the same files, and compute nodes often lack internet access anyway).
# Run once, BEFORE submitting, on a node with internet access:
#   bash scripts/setup_cluster.sh                    # datasets + timm PTM weights
#   python scripts/gen_final_vision_configs.py
#
# >>> CLUSTER-SPECIFIC: conda env name/activation -- adjust for your cluster <<<
# Set CONDA_ENV_NAME if your env isn't named "treelora"; uses `conda shell.bash
# hook` (not a hardcoded install path) so it works regardless of where conda
# is installed, as long as `conda` is on PATH (e.g. via `module load`).

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (LAMDA-PILOT/)

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

mkdir -p run_logs/final_vision

mapfile -t ALL_TAGS < <(ls exps/final_vision/*.json | xargs -n1 basename | sed 's/\.json$//' | sort)
TAG="${ALL_TAGS[$SLURM_ARRAY_TASK_ID]}"

echo "=== array task ${SLURM_ARRAY_TASK_ID}: ${TAG} ==="
python main.py --config "exps/final_vision/${TAG}.json" \
  > "run_logs/final_vision/${TAG}.out" 2>&1
echo "=== finished ${TAG} (exit $?) ==="
