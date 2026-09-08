#!/bin/bash
#SBATCH --job-name=omnibenchmark1k_200t
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/omnibenchmark1k_200t/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/omnibenchmark1k_200t/slurm_%j.err
#
# Long-sequence comparison (2026-09-08 user request): all 8 established
# methods (SeqLoRA, SketchLoRA [orth, the standing default], O-LoRA, InfLoRA,
# CL-LoRA, RainbowPrompt, TUNA, EASE) on a 200-task OmniBenchmark-1k split
# (init_cls=5, increment=5 -- half the classes/task of the usual 100-task
# split, same 1000 total classes), 3 seeds each = 24 runs. "How does each
# method hold up over a much longer task sequence?"
#
# Each method's config is its own most-recent/best-established
# OmniBenchmark-1k config verbatim (dataset-split fields overridden only) --
# see scripts/gen_omnibenchmark1k_200t_configs.py's own docstring for the
# exact source file per method and why (wave1_final vs wave1_final_completion
# vs the dedicated orth-SketchLoRA campaign -- cllora/ease/tuna specifically
# carry the SETTLED post-LR-sweep learning rates for OmniBenchmark-1k,
# verified present in the source files before this script was written, not
# assumed).
#
# SINGLE GPU BY CONSTRUCTION (this cluster only has one) -- same plain
# sequential pattern as every other single-GPU campaign here.
#
# WALL TIME (96h): this is a LONG campaign -- CL-LoRA/EASE/TUNA alone took
# multiple hours EACH at the usual 100-task split in earlier campaigns
# (ce_final_all), and 200 tasks roughly doubles the task-boundary/eval
# overhead per run (same total training compute, ~2x the checkpoints). 24
# runs sequentially on one GPU will almost certainly need several
# resubmissions; relies on the RESUMABLE design below, same as every other
# campaign here, to make partial progress across them.
#
# ORDER: method-outer, seed-inner (all 3 seeds of one method before moving to
# the next) so a resubmission that runs out of time still banks COMPLETE
# 3-seed mean+-std for as many methods as possible. Roughly fastest-to-
# slowest per this project's own prior observations at 100 tasks (SeqLoRA/
# InfLoRA/SketchLoRA/O-LoRA/RainbowPrompt fast-to-moderate; TUNA/CL-LoRA/EASE
# each took multiple hours per single run) -- not re-measured at 200 tasks,
# just the best available prior.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <model_name>/metrics_<tag>.json's own status=="done" -- this campaign's
# own prefix namespaces every tag, so it can only ever match a genuine
# attempt from THIS campaign).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=omnibenchmark1k_200t
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
python scripts/gen_omnibenchmark1k_200t_configs.py
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

echo "==== ${CAMPAIGN} start: 8 methods x 3 seeds, single GPU $(date) ===="

# method-outer, seed-inner, roughly fastest-to-slowest -- see header note.
METHODS="seqlora inflora sketchlora olora rainbowprompt tuna cllora ease"
SEEDS="1993 1996 1999"

for METHOD in $METHODS; do
  for SEED in $SEEDS; do
    run_one "$CONFIG_DIR/${METHOD}_s${SEED}.json"
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
