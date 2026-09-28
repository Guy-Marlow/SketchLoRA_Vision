#!/bin/bash
#SBATCH --job-name=freqdir_dense_wave1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_dense_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/freqdir_dense_wave1/slurm_%j.err
#
# freqdir_dense merge_op campaign, wave 1 (2026-09-28 user request) -- FD
# shrinkage on the randomized-SVD probe of the dense composite update (see
# utils/freqdir_dense.py, models/sketchlora.py's merge_op="freqdir_dense"
# branch), vs. the standing SketchLoRA (orth, merge_op="randsvd") baseline,
# on 3 benchmarks x 3 seeds = 9 runs: CIFAR-100/cifar224-10t,
# ImageNet-R-20t, OmniBenchmark-1k-100t. See
# scripts/gen_freqdir_dense_wave1_configs.py's own docstring for the exact
# source configs each run is copied from.
#
# 24h budget: 9 runs sequentially at this project's own measured per-run
# wall time for these benchmarks (ImageNet-R/CIFAR-100 SketchLoRA runs
# ~2.5-3h each, OmniBenchmark-1k ~2.9h including eval -- see the
# sketchlora_orth_wave1_datasets / wave1_final fixedrank_orth05 metrics)
# comes to roughly 9 * 3h =~ 27h at the high end -- may not finish in one
# submission once per-run variance is accounted for. Relies on the
# RESUMABLE design below to make partial progress across resubmissions, same
# as every other long campaign here -- not a bug, expected.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as
# scripts/freqdir_wave1.slurm.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align and this campaign's own prefix, so it
# can only ever match a genuine attempt from THIS campaign).
#
# ALL OUTPUT under run_logs/freqdir_dense_wave1/ (.out logs here; sketch_diag
# under its own <dataset>/diag/ subdir -- see the config generator;
# final_metrics land at the usual project-wide run_logs/final/ location,
# same as every other oracle-path campaign).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=freqdir_dense_wave1
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
# same idiom as scripts/freqdir_wave1.slurm.
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
python scripts/gen_freqdir_dense_wave1_configs.py
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

echo "==== ${CAMPAIGN} start: 9 runs, single GPU $(date) ===="

while IFS= read -r config; do
  run_one "$config"
done < <(find "$CONFIG_DIR" -name "*.json" | sort)

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
