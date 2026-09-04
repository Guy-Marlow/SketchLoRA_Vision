#!/bin/bash
#SBATCH --job-name=sketchlora_orth_sens_ablations
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_sens_ablations/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_orth_sens_ablations/slurm_%j.err
#
# SketchLoRA orth sensitivity + ablation campaign (2026-09-02 user request).
# 69 runs total (scripts/gen_sketchlora_orth_sens_ablations_configs.py's own
# docstring has the exact breakdown): a 9-point (align_weight, svd_rank)
# sensitivity grid on ImageNet-R-20t (8 cells -- the 9th, w=0.5/r=10, reuses
# an already-completed result) and CIFAR-100/cifar224-10t (all 9 cells),
# each x 3 seeds; plus 3 ablations (countsketch+orth, exactsvd+orth,
# random-rank-selection+orth) x 2 datasets x 3 seeds. The fixed-rank NON-orth
# baseline is intentionally excluded -- already have that result on both
# datasets.
#
# >>> GPU COUNT / RESOURCES: --gres=gpu:2 / 20 cpu / 96G below is a GUESS,
# scaled off scripts/sketchlora_ablations_imagenetr20t.slurm's own known-good
# 1-GPU/10-cpu/48G request (roughly 2x, matching 2 GPUs) -- NOT verified
# against this cluster's actual H200 allocation/queue limits, since this
# script was written without cluster access. Adjust to whatever the H200
# partition actually grants before submitting. NGPU below is DERIVED from
# torch.cuda.device_count() regardless (same convention as that precedent
# script), so the worker-pool fan-out always matches whatever --gres actually
# grants -- no other line needs editing if the GPU count above changes. <<<
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align, so this campaign's own prefix
# namespaces every tag; can only ever match a genuine attempt from THIS
# campaign, never collide with an unrelated run).
#
# ALL OUTPUT under run_logs/sketchlora_orth_sens_ablations/ (.out logs here;
# sketch_diag under its own diag/ subdir per (sensitivity|ablations)/<dataset>/
# -- see the config generator; final_metrics/CE2 land at their usual
# project-wide run_logs/final/ and run_logs/ce2/ locations, same as every
# other oracle-path campaign -- ce2_enabled is NOT set here, this campaign is
# about accuracy/forgetting/rank behavior, not CE profiling).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_orth_sens_ablations
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

NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>/tmp/${CAMPAIGN}_preflight_$$.err)
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat /tmp/${CAMPAIGN}_preflight_$$.err
  rm -f /tmp/${CAMPAIGN}_preflight_$$.err
  exit 1
fi
rm -f /tmp/${CAMPAIGN}_preflight_$$.err
echo "==== detected NGPU=${NGPU} (torch.cuda.device_count(), derived not hardcoded) $(date) ===="

echo "==== config generation start $(date) ===="
python scripts/gen_sketchlora_orth_sens_ablations_configs.py
echo "==== config generation done $(date) ===="

echo "==== ${CAMPAIGN} start, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=config path
  local gpu="$1" config="$2"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sensitivity/cifar224
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) [gpu ${gpu}] ${subdir}/${tag} $(date) ===="
    return
  fi
  if [ -f "$outfile" ]; then
    mv "$outfile" "${outfile}.prev-${SLURM_JOB_ID:-unknown}"
  fi
  echo "==== START [gpu ${gpu}] ${subdir}/${tag} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${subdir}/${tag} $(date) ===="
}

JOBS_FIFO="${OUT_DIR}/.jobs.fifo"
rm -f "$JOBS_FIFO"
mkfifo "$JOBS_FIFO"

(
  find "$CONFIG_DIR" -name "*.json" | sort
) > "$JOBS_FIFO" &
FIFO_WRITER_PID=$!

worker() {
  local gpu="$1"
  local config
  while IFS= read -r config; do
    run_one "$gpu" "$config"
  done < "$JOBS_FIFO"
}

for (( g=0; g<NGPU; g++ )); do
  worker "$g" &
done
wait
wait "$FIFO_WRITER_PID" 2>/dev/null || true
rm -f "$JOBS_FIFO"

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
