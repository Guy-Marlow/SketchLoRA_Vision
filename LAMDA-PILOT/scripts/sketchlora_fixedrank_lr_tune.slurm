#!/bin/bash
#SBATCH --job-name=sketchlora_fixedrank_lr_tune
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_fixedrank_lr_tune/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_fixedrank_lr_tune/slurm_%j.err
#
# SketchLoRA fixed-rank learning-rate tuning sweep (2026-08-18 user request).
# Motivation: O-LoRA's orthogonality penalty against a frozen adapter
# effectively brakes the live slot's updates (see bankcap_olora_mechanism_
# finding memory) -- raising the question of whether SketchLoRA's own lr
# (uniformly 0.001 across every wave1_final benchmark, see
# other_methods_config_and_lr_sweep memory) is simply mistuned, on ImageNet-R
# and possibly elsewhere. This sweeps 6 learning rates x 2 seeds (1993/1996)
# x 3 benchmarks = 36 runs for the FIXED-RANK variant specifically (no
# adaptive rank growth via svd_energy_target -- isolates the lr question from
# the rank-growth mechanism).
#
# Base template: exps/sketchlora_ablations_imagenetr20t/sketchlora_fixedrank_
# s1993.json, copied verbatim by scripts/gen_sketchlora_fixedrank_lr_tune_
# configs.py -- randSVD merge_op, fixed rank 10 (svd_rank=10, no
# svd_energy_target, sketchlora_admission=global_eps), batch_size=48,
# tuned_epoch=20, weight_decay=0.0005 on head+live adapter with
# sketchlora_lora_wd=0.0 zeroing it for the sketch specifically -- ALL
# unchanged from that file per explicit user instruction. Only dataset/
# init_cls/increment/stop_after_tasks/seed/init_lr/prefix vary per cell.
#
# LEARNING RATES: 1e-2, 3e-3, 1e-3, 8e-4, 3e-4, 1e-4 (explicit user list).
#
# TASK COUNTS (stop_after_tasks, explicit user spec): CIFAR-100-10t -> 5,
# ImageNet-R-20t -> 10, OmniBenchmark-1k-100t -> 20. init_cls/increment=10/10
# for all three, matching the wave1_final split convention.
#
# NO STRUCTURED LOGGING (final_metrics=false, print_forget=false, matching
# scripts/gen_lr_sweep_configs.py's established convention for a pure lr
# sweep) -- trainer.py prints "CNN top5 curve:"/"Average Accuracy (CNN):"
# unconditionally, so the number this sweep needs is fully recoverable from
# each .out log with no MetricsLogger/CE-ledger overhead.
#
# RESUMABILITY -- deliberately NOT the bare `grep -q "Average Accuracy"`
# pattern scripts/cllora_ease_tuna_rainbow_sens.slurm uses: that line prints
# once per TASK, not once per run (confirmed this session against wave1_rerun_
# treelora's own .out logs), so a wall-time-killed run after task 1 would
# already contain that string and be wrongly marked done forever. Instead,
# run_one() counts how many "CNN top5 curve:" lines exist and compares
# against that cell's own stop_after_tasks -- a run is "done" only once it
# has printed exactly that many checkpoint lines.
#
# WALL TIME -- FLAGGED, READ BEFORE ASSUMING THIS FITS IN ONE SUBMISSION.
# Real per-task cost at rank=10/batch=48/20-epoch on this project's shared
# LoRA scaffold has run ~9-10 min/task on ImageNet-R this session (TreeLoRA's
# routing-fix diagnostic, same epoch/batch/rank settings). Scaling that
# per-task estimate (SketchLoRA's per-boundary SVD compression adds some
# overhead on top, not measured separately here):
#   cifar224  (5 tasks):  ~50 min/run  x 6 lr x 2 seed = ~10h
#   imagenetr (10 tasks): ~100 min/run x 6 lr x 2 seed = ~20h
#   omnibenchmark1k (20 tasks): ~200-300 min/run x 6 lr x 2 seed = ~40-50h
# Total estimate: ~70-80h against a 30h budget (bumped from an initial 20h
# per explicit user request) -- this will still need several resubmissions to
# finish all 36 cells; the resumable design above is what makes that safe,
# not just convenient.
#
# RUN ORDER -- SEED-MAJOR (explicit user request): seed is the OUTERMOST
# loop, so every (dataset x lr) cell for seed 1993 runs to completion before
# seed 1996 starts at all. This guarantees that if the campaign never gets to
# the second seed, what exists is a COMPLETE grid -- every benchmark, every
# learning rate -- for one seed, rather than a partial mix of both seeds on
# just the cheaper benchmarks. Dataset is still ordered cheapest-first
# (cifar224 -> imagenetr -> omnibenchmark1k) within each seed, so a wall-time
# cutoff mid-seed still secures the cheap benchmarks' full lr curves first.
#
# SINGLE GPU BY CONSTRUCTION: plain sequential loop, CUDA_VISIBLE_DEVICES
# pinned to 0, no worker-pool/FIFO fan-out.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=sketchlora_fixedrank_lr_tune
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
# same idiom as every other script in this project.
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
python scripts/gen_sketchlora_fixedrank_lr_tune_configs.py
echo "==== config generation done $(date) ===="

LR_TAGS=(1e-2 3e-3 1e-3 8e-4 3e-4 1e-4)
SEEDS=(1993 1996)

# dataset -> expected stop_after_tasks (must match gen_sketchlora_fixedrank_
# lr_tune_configs.py's DATASETS table exactly).
declare -A EXPECTED_TASKS=( [cifar224]=5 [imagenetr]=10 [omnibenchmark1k]=20 )
# cheapest-first, see WALL TIME note above.
DATASETS=(cifar224 imagenetr omnibenchmark1k)

run_one() {
  # $1=dataset $2=lr_tag $3=seed
  local dataset="$1" lr_tag="$2" seed="$3"
  local tag="sketchlora_${dataset}_${lr_tag}_s${seed}"
  local config="${CONFIG_DIR}/${tag}.json"
  local outfile="${OUT_DIR}/${tag}.out"
  local expected="${EXPECTED_TASKS[$dataset]}"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  # Completion check: count "CNN top5 curve:" lines, NOT a bare grep -q for
  # "Average Accuracy (CNN):" -- that line (and the top5-curve line) print
  # once per TASK, not once per run, so a wall-time-killed run after task 1
  # would already contain the string and be wrongly marked done forever. Only
  # a run whose .out has exactly `expected` such lines has actually finished
  # every configured task.
  if [ -f "$outfile" ]; then
    local n_done
    n_done=$(grep -c "CNN top5 curve:" "$outfile" 2>/dev/null || echo 0)
    if [ "$n_done" -ge "$expected" ]; then
      echo "==== SKIP (done, ${n_done}/${expected} checkpoints) ${tag} $(date) ===="
      return
    fi
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

TOTAL=$(( ${#DATASETS[@]} * ${#LR_TAGS[@]} * ${#SEEDS[@]} ))
echo "==== ${CAMPAIGN} start: ${TOTAL} runs (3 datasets x 6 lrs x 2 seeds), single GPU $(date) ===="

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for LR_TAG in "${LR_TAGS[@]}"; do
      run_one "$DATASET" "$LR_TAG" "$SEED"
    done
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
