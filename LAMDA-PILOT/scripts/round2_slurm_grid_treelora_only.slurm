#!/bin/bash
#SBATCH --job-name=round2_slurm_grid_treelora_only
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_treelora_only_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_treelora_only_%j.err
#
# TreeLoRA-only rerun of the round2_slurm_grid campaign (2026-07-31), same
# naming scheme, config directory, and output paths as scripts/
# round2_slurm_grid.slurm -- writes/overwrites ONLY treelora_{100,200}mb_
# s{1993,1996,1999}.out (6 files), never touches seqlora/olora/inflora/
# sketchlora output.
#
# WHY THIS EXISTS: every one of the original grid's 6 TreeLoRA cells crashed
# on its very first bounded-memory training cycle --
#   AttributeError: 'NoneType' object has no attribute 'shape'
#   (utils/kd_tree.py:104, end_task, called from models/treelora.py's
#   _stream_end_chunk -> self.tree.end_task(self._stream_chunk))
# Root cause: at the commit the original grid ran under (d964a0e, tip of
# 'main' as of 2026-07-27 03:40), models/treelora.py had no _bounded_memory_
# mixin _bounded_train_epoch override, so BoundedMemoryMixin's own generic
# training loop ran instead -- it only offers the narrow _stream_extra_loss
# hook (models/stream_mixin.py's convention), which cannot drive TreeLoRA's
# tree regularizer (new_epoch_init/step/insert_grad/tree_search/get_loss).
# self.tree.current_grad was therefore never populated via insert_grad(), so
# end_task's `self.current_grad.shape[0]` crashed on cycle 0 for every seed
# and every budget, identically (confirmed: all 6 .out files are the same
# 52-line traceback, differing only in seed-derived class order / timestamps).
# A _bounded_train_epoch override for TreeLoRA (mirroring the existing
# _stream_train_epoch override's tree-regularizer wiring, but with bounded_
# memory's full-logit + cycle-class-mask convention instead of stream_run's
# local [lo,hi) slicing) now exists in models/treelora.py -- this script reruns
# only the 6 cells that never produced a result, against that fixed code.
#
# NOT rerun: seqlora_100mb_s1993 and sketchlora_100mb_s1993, which are
# separately incomplete (killed job 21982, never reissued by the successful
# resubmission 21983 due to an unrelated job-queue race) -- out of scope for
# this script, which is TreeLoRA-only by design; rerun those two separately
# with the original round2_slurm_grid.slurm's run_one() convention if needed.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/round2_slurm_grid

# Refuse to run alongside the FULL grid script (both would claim GPUs 0/1) --
# check its lockfile too, not just our own.
MAIN_LOCKFILE="run_logs/round2_slurm_grid/.round2_slurm_grid.lock"
if [ -f "$MAIN_LOCKFILE" ] && kill -0 "$(cat "$MAIN_LOCKFILE")" 2>/dev/null; then
  echo "FATAL: scripts/round2_slurm_grid.slurm (PID $(cat "$MAIN_LOCKFILE")) appears to be running -- refusing to start a second job on the same 2 GPUs. Wait for it to finish or delete $MAIN_LOCKFILE if stale."
  exit 1
fi
LOCKFILE="run_logs/round2_slurm_grid/.round2_slurm_grid_treelora_only.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another round2_slurm_grid_treelora_only.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check (same convention as round2_slurm_grid.slurm).
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/round2_slurm_grid_treelora_only_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/round2_slurm_grid_treelora_only_preflight_$$.err
  rm -f /tmp/round2_slurm_grid_treelora_only_preflight_$$.err
  exit 1
fi
rm -f /tmp/round2_slurm_grid_treelora_only_preflight_$$.err

# Configs already exist and are already correct (committed, unchanged since
# the original grid: exps/round2_slurm_grid/treelora_{100,200}mb_s{1993,1996,
# 1999}.json) -- verify presence rather than regenerating all 30 of the full
# grid's configs, to keep this script's write footprint to TreeLoRA only.
BUDGETS=(100 200)
SEEDS=(1993 1996 1999)
MISSING=0
for BUDGET in "${BUDGETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    cfg="exps/round2_slurm_grid/treelora_${BUDGET}mb_s${SEED}.json"
    if [ ! -f "$cfg" ]; then
      echo "FATAL: missing expected config $cfg -- run scripts/gen_round2_slurm_grid_configs.py first (regenerates the full 30-config grid, all methods)."
      MISSING=1
    fi
  done
done
[ "$MISSING" -eq 1 ] && exit 1

NGPU=2
echo "==== round2_slurm_grid_treelora_only start: ${#BUDGETS[@]} budgets x 1 method (treelora) x ${#SEEDS[@]} seeds, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=budget $3=seed -- identical naming/skip-check convention to
  # round2_slurm_grid.slurm's run_one(), method hardcoded to treelora.
  local gpu="$1" budget="$2" seed="$3" method="treelora"
  local tag="${method}_${budget}mb_s${seed}"
  local config="exps/round2_slurm_grid/${tag}.json"
  local prefix="round2_slurm_grid_omni30t_${budget}mb_${method}_s${seed}"
  local json="run_logs/boundedmem_${method}_${prefix}_s${seed}.json"
  local metrics_json="run_logs/final/${method}/metrics_${method}_${prefix}_s${seed}.json"

  if [ -f "$json" ] && grep -q '"partial": false' "$json" \
     && [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done, accuracy+metrics both complete) ${tag} $(date) ===="
    return
  fi
  echo "==== START [gpu ${gpu}] ${tag} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --config "$config" \
    > "run_logs/round2_slurm_grid/${tag}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${tag} $(date) ===="
}

for BUDGET in "${BUDGETS[@]}"; do
  echo "==== BUDGET ${BUDGET}MB (treelora only) start $(date) ===="

  JOBS_FIFO="run_logs/round2_slurm_grid/.jobs_treelora_only_${BUDGET}mb.fifo"
  rm -f "$JOBS_FIFO"
  mkfifo "$JOBS_FIFO"

  ( for SEED in "${SEEDS[@]}"; do echo "${SEED}"; done ) > "$JOBS_FIFO" &
  FIFO_WRITER_PID=$!

  worker() {
    local gpu="$1" seed
    while read -r seed; do
      run_one "$gpu" "$BUDGET" "$seed"
    done < "$JOBS_FIFO"
  }

  for (( g=0; g<NGPU; g++ )); do
    worker "$g" &
  done
  wait
  wait "$FIFO_WRITER_PID" 2>/dev/null || true
  rm -f "$JOBS_FIFO"

  echo "==== BUDGET ${BUDGET}MB (treelora only) done $(date) ===="
done
echo "==== ALL TREELORA RUNS COMPLETE (or wall reached) $(date) ===="
