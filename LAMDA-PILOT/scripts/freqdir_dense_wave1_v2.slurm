#!/bin/bash
#SBATCH --job-name=freqdir_dense_wave1_v2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_dense_wave1_v2/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_dense_wave1_v2/slurm_%j.err
#
# freqdir_dense merge_op campaign, CURRENT (randomized-probe, window-
# restricted) implementation -- 2026-09-30 local run on the A5000 (6/6
# complete, see run_logs/freqdir_dense_wave1_v2/ locally), re-run here on
# the cluster's larger GPUs for robustness/a second hardware's numerics.
# CIFAR-100/cifar224-10t + ImageNet-R-20t, 3 seeds each = 6 runs. See
# scripts/gen_freqdir_dense_wave1_v2_configs.py's own docstring for exactly
# why this is a new campaign name (not freqdir_dense_wave1 -- that's the
# ORIGINAL, verified-broken randomized-probe implementation with no window
# restriction; not freqdir_dense_wave1_exactsvd -- that's a different
# algorithm, exact SVD, that briefly existed at commit beb04c8 before being
# superseded). This uses HEAD's implementation: randomized SVD probe,
# shrinkage restricted to S[:composite_rank] to exclude the oversampling-
# padding noise floor (models/sketchlora.py's merge_op=="freqdir_dense"
# branch, utils/freqdir_dense.py, commit 1a6ae6f).
#
# Local A5000 results (3 seeds each, final-task Top-1%/Top-5%):
#   CIFAR-100-10t:  83.62+-1.22 / 97.75+-0.09
#   ImageNet-R-20t: 55.95+-1.28 / 77.00+-1.03
# vs. the plain-randsvd orthogonalized-SketchLoRA baseline (same config
# otherwise): CIFAR-100-10t 84.14+-1.46 (freqdir_dense ~even), ImageNet-R-20t
# 58.80+-0.76 (freqdir_dense notably worse, outside std bands). This cluster
# run exists to confirm that result isn't an A5000-specific numerical
# artifact before treating it as a real finding.
#
# 24h budget: 6 runs sequentially, each comparable in cost to plain randsvd
# SketchLoRA (freqdir_dense's shrinkage step is cheap relative to a full
# ViT-B/16 training run) -- the local A5000 runs averaged ~3.25h/run
# (19.5h / 6 runs), so 24h is generous headroom on faster cluster GPUs.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as
# scripts/freqdir_dense_wave1.slurm / scripts/sketchlora_orth_sens_
# omnibenchmark1k.slurm.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align and this campaign's own prefix, so it
# can only ever match a genuine attempt from THIS campaign).
#
# ALL OUTPUT under run_logs/freqdir_dense_wave1_v2/ (.out logs here;
# sketch_diag under its own <dataset>/diag/ subdir -- see the config
# generator; final_metrics land at the usual project-wide run_logs/final/
# location, same as every other oracle-path campaign).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=freqdir_dense_wave1_v2
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
# same idiom as every other campaign .slurm in this project.
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
python scripts/gen_freqdir_dense_wave1_v2_configs.py
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

echo "==== ${CAMPAIGN} start: 6 runs, single GPU $(date) ===="

while IFS= read -r config; do
  run_one "$config"
done < <(find "$CONFIG_DIR" -name "*.json" | sort)

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
