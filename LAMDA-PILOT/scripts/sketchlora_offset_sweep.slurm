#!/bin/bash
#SBATCH --job-name=sketchlora_offset_sweep
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_offset_sweep/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_offset_sweep/slurm_%j.err
#
# Epoch-offset consolidation sweep for vanilla SketchLoRA (2026-11-06 user
# request): model_name=sketchlora_align, merge_op="qrsvd",
# sketchlora_align_mode="orth" (this project's current canonical
# "SketchLoRA" -- see memory sketchlora-orth-is-now-default-convention),
# under boundary_mode="offset_consolidation" (models/stream_mixin.py::
# offset_consolidation_run) instead of the ordinary oracle per-task loop.
#
# offset_k held at 20 (== tuned_epoch, a fixed epoch count per segment,
# NOT swept); offset_phi swept across {1,2,3,4,5}. 2 datasets (CIFAR-100-
# 10t, ImageNet-R-20t) x 5 offsets x 3 seeds (1993/1996/1999) = 30 runs.
# Full-length runs, no stop_after_tasks.
#
# Config generator: scripts/gen_sketchlora_offset_sweep_configs.py (run at
# the top of this script, same as every other campaign here -- configs are
# regenerated, not hand-committed, so a resubmission always starts from a
# known-identical set of 30 files).
#
# LOGGING/DIAGNOSTICS kept: print_forget=true (genuinely supported now --
# 2026-11-06 fix added a real forgetting computation to
# offset_consolidation_run/._stream_eval, which previously had NONE under
# ANY boundary_mode streaming driver in this project -- see that method's
# own docstring), sketch_diag=true + a per-dataset sketchlora_diag_dir
# (run_logs/sketchlora_offset_sweep/<dataset>/diag -- unique to this
# campaign, can't collide with any other campaign's sketch diagnostics),
# final_metrics=false (matches EVERY other boundary_mode streaming config
# in this project -- that machinery is oracle-loop-only and is simply never
# reached under _run_stream, so leaving it true would be a silent no-op,
# not extra data).
#
# RESUMABLE via scripts/_stream_run_done.py <config> offset (checks
# run_logs/offset_<model>_<prefix>_s<seed>.json's own partial==False --
# the streaming-path analog of scripts/_bankcap_run_done.py, which does NOT
# apply here since final_metrics=false means it would always report "not
# done" and defeat resumability entirely).
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

CAMPAIGN=sketchlora_offset_sweep
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
python scripts/gen_sketchlora_offset_sweep_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. cifar224
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_stream_run_done.py "$config" offset 2>/dev/null; then
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

# Run order: cifar224 first (cheaper, 10 tasks -- faster feedback), then
# imagenetr (20 tasks). `sort` within each dataset keeps offset/seed order
# deterministic (phi1..phi5, each x s1993/s1996/s1999).
echo "==== ${CAMPAIGN} start: 30 runs, single GPU $(date) ===="

for dataset in cifar224 imagenetr; do
  while IFS= read -r config; do
    run_one "$config"
  done < <(find "$CONFIG_DIR/$dataset" -name "*.json" 2>/dev/null | sort)
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
