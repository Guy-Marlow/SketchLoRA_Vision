#!/bin/bash
#SBATCH --job-name=sketchlora_orth_sens_ablations
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
# SINGLE GPU BY CONSTRUCTION (2026-09-04 correction -- this cluster only has
# one GPU available; an earlier draft of this script guessed at 2 and used a
# multi-GPU worker-pool/FIFO fan-out that assumed it). Rewritten to the
# plain sequential single-GPU pattern already proven on this project's other
# campaigns (scripts/ce_final_all.slurm / ce_final_all_retry3.slurm) --
# CUDA_VISIBLE_DEVICES pinned to 0, one run_one() loop, no FIFO, no worker
# function. --gres=gpu:1 / --cpus-per-task=8 / --mem=48G, same resource
# request as scripts/sketchlora_ablations_imagenetr20t.slurm's own known-good
# 1-GPU config for a comparable SketchLoRA campaign (not a guess this time).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# sketchlora_align/metrics_<tag>.json's own status=="done" -- every config
# here uses model_name=sketchlora_align, so this campaign's own prefix
# namespaces every tag; can only ever match a genuine attempt from THIS
# campaign, never collide with an unrelated run).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's log
# -- run_one() below backs up any existing outfile first (same convention as
# ce_final_all.slurm).
#
# ALL OUTPUT under run_logs/sketchlora_orth_sens_ablations/ (.out logs here;
# sketch_diag under its own diag/ subdir per (sensitivity|ablations)/<dataset>/
# -- see the config generator; final_metrics/CE2 land at their usual
# project-wide run_logs/final/ and run_logs/ce2/ locations, same as every
# other oracle-path campaign -- ce2_enabled is NOT set here, this campaign is
# about accuracy/forgetting/rank behavior, not CE profiling).
#
# 48h wall time for 69 runs sequentially on one GPU will almost certainly NOT
# finish in one submission (SketchLoRA/ImageNet-R runs alone have taken
# ~2.5-3h each in this project's other campaigns; CIFAR-100/cifar224's 10-task
# split is shorter). Relies on the RESUMABLE design above to make partial
# progress across repeated resubmissions, same as every other long campaign
# here -- not a bug, expected.
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
python scripts/gen_sketchlora_orth_sens_ablations_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sensitivity/cifar224
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

echo "==== ${CAMPAIGN} start: 69 runs, single GPU $(date) ===="

while IFS= read -r config; do
  run_one "$config"
done < <(find "$CONFIG_DIR" -name "*.json" | sort)

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
