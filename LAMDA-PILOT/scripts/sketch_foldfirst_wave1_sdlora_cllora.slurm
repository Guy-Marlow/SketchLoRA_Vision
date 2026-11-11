#!/bin/bash
#SBATCH --job-name=sketch_foldfirst_wave1_sdlora_cllora
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketch_foldfirst_wave1_sdlora_cllora/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketch_foldfirst_wave1_sdlora_cllora/slurm_%j.err
#
# SD-LoRA-Sketch-FoldFirst + CL-LoRA-Sketch-FoldFirst only (2026-11-11 user
# request) -- a deliberate SUBSET of the original scripts/sketch_foldfirst_
# wave1.slurm campaign (commit ed10e84). LoRA-Subtraction-Sketch-FoldFirst
# is EXCLUDED here on purpose: its underlying code (backbone/vit_lorasub.py,
# models/lorasub.py, models/lorasub_sketch.py) has a separate set of
# DRS-probe/init/prototype fixes that are still being smoke-tested locally
# as of this script's creation, not yet committed or pushed. It will be
# re-run later, together with plain (growing-bank) LoRA-Subtraction, under
# its own separate SLURM file once those fixes are validated and pushed --
# do NOT add lorasub_sketch_foldfirst back into this script's run loop to
# "complete" it; that is intentionally a different campaign submission.
#
# 2 methods x 5 datasets (CIFAR-100-10t, ImageNet-R-20t,
# OmniBenchmark-1k-100t, SUN397-10t, Food-101-20t) x 3 seeds
# (1993/1996/1999) = 30 runs. Both methods were already smoke-tested clean
# (3-task CIFAR-100) and pushed before this script was written -- see
# models/sdlora_sketch_foldfirst.py / models/cllora_sketch_foldfirst.py's
# own module docstrings for the fold-before-eval mechanism and why it's
# correct.
#
# HYPERPARAMETERS: imported directly from scripts/gen_sketch_foldfirst_
# wave1_configs.py (not re-derived) via scripts/gen_sketch_foldfirst_wave1_
# sdlora_cllora_configs.py, confirmed byte-identical to what the original
# campaign would generate for these two methods. `prefix` or deliberately
# unchanged from the original convention (`sketch_foldfirst_wave1_...`) --
# this is genuinely the same logical campaign, just split across two SLURM
# submissions, so results land under the same final metrics naming either
# way.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <model_name>/metrics_<tag>.json's own status=="done") -- these are all
# ordinary oracle-mode runs, final_metrics=true throughout.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every other
# campaign script in this project (this cluster has exactly one GPU
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

CAMPAIGN=sketch_foldfirst_wave1_sdlora_cllora
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
python scripts/gen_sketch_foldfirst_wave1_sdlora_cllora_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sdlora_sketch_foldfirst/cifar224
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
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

# Run order: method-major, dataset-minor, seed order deterministic within
# each (same convention as scripts/sketch_foldfirst_wave1.slurm). NOTE:
# lorasub_sketch_foldfirst is deliberately NOT in this list -- see header.
echo "==== ${CAMPAIGN} start: 30 runs, single GPU $(date) ===="

for method in sdlora_sketch_foldfirst cllora_sketch_foldfirst; do
  for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
    while IFS= read -r config; do
      run_one "$config"
    done < <(find "$CONFIG_DIR/$method/$dataset" -name "*.json" 2>/dev/null | sort)
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
