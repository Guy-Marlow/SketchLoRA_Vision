#!/bin/bash
#SBATCH --job-name=round2_slurm_grid
#SBATCH --time=35:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_%j.err
#
# Round-2 SLURM grid (user request 2026-07-27, scheduling revised 2026-07-27):
# SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA x {100, 200}MB x 3 seeds
# (1993, 1996, 1999), OmniBenchmark-1K, FULL 100-task split (REVISED
# 2026-07-27 from a 30-task truncation -- see
# scripts/gen_round2_slurm_grid_configs.py), bounded_memory boundary mode --
# 30 runs total. SketchLoRA uses svd_energy_target=0.01 and
# merge_op=randsvd/svd_oversampling=10 (REVERTED to RandSVD, user-specified:
# RandSVD is central to the method's guarantees and its incurred error is
# low -- supersedes round-2 2.5's temporary exact-SVD-until-cleared fallback,
# see scripts/gen_round2_slurm_grid_configs.py). InfLoRA uses a CONSTANT
# lamb=lame=0.975 retention threshold (removes InfLoRA's total-stream-length
# dependence -- see docs/plan_c_bounded_memory_round2.md's sensitivity-run
# writeup for why), replacing the default 0.95->1.0 ramp for every InfLoRA
# run in this grid.
#
# BUDGET-MAJOR (100MB fully drains before 200MB starts), SEED-MAJOR-CYCLING
# WITHIN a budget (user-specified 2026-07-27, supersedes this script's
# earlier method-major-with-synchronized-lanes design): for a given budget,
# build ONE FLAT QUEUE of 15 jobs = 3 seeds x 5 methods, methods always in
# order [seqlora, sketchlora, olora, inflora, treelora] within each seed's
# block, seed 1996's block only after ALL of seed 1993's 5 methods are
# QUEUED (not necessarily finished -- see below), then seed 1999's block.
# Two GPU workers consume this SAME shared queue: initially GPU 0 takes
# seqlora(1993) and GPU 1 takes sketchlora(1993) (the queue's first two
# items); from then on, whichever GPU frees up next simply claims the NEXT
# unclaimed queue item, with NO per-method or per-seed synchronization
# barrier -- so the actual completion cascade is: first GPU to finish moves
# to olora(1993), the next to finish moves to inflora(1993), then
# treelora(1993), then seqlora(1996), and so on, exactly cycling through
# every method once per seed before any method starts its next seed. This is
# a genuine shared work-queue (via a FIFO both workers `read` from), NOT the
# static odd/even round-robin split scripts/final_vision.slurm uses -- that
# static split pre-assigns items by parity regardless of real finish order,
# which does not reproduce "whichever finishes first gets replaced by the
# next method" behavior.
#
# Every config uses "device": ["0"] uniformly (see
# scripts/gen_round2_slurm_grid_configs.py) -- physical GPU selection happens
# entirely via CUDA_VISIBLE_DEVICES per worker lane below, sidestepping the
# CUDA_DEVICE_ORDER/PCI-bus device-index gotcha hit earlier this session
# (device index 4 silently resolving to the small DGX display card without
# PCI_BUS_ID ordering) -- CUDA_VISIBLE_DEVICES restricts visibility before
# that ambiguity can occur.
#
# RESUMABLE: skips any run whose run_logs/boundedmem_*.json already has
# "partial": false (trainer.py's clean-finish marker for bounded_memory runs,
# distinct from final_vision.slurm's own "status": "done" convention used by
# its non-streaming grid) -- re-submitting after a 35h wall-kill picks up
# exactly where it left off (any job that was mid-run when the wall hit gets
# re-queued at that job's original position, since the FIFO is rebuilt fresh
# from the full method x seed list every submission and skip-if-done drops
# whatever already finished).
#
# LOCKFILE: refuses to start a second instance while one is already running.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/round2_slurm_grid

LOCKFILE="run_logs/round2_slurm_grid/.round2_slurm_grid.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another round2_slurm_grid.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check -- fail fast and loudly rather than silently
# burning the wall-clock budget on 30 ImportErrors swallowed by `|| true`.
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/round2_slurm_grid_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/round2_slurm_grid_preflight_$$.err
  rm -f /tmp/round2_slurm_grid_preflight_$$.err
  exit 1
fi
rm -f /tmp/round2_slurm_grid_preflight_$$.err

echo "==== config generation start $(date) ===="
python scripts/gen_round2_slurm_grid_configs.py
echo "==== config generation done $(date) ===="

BUDGETS=(100 200)
METHODS=(seqlora sketchlora olora inflora treelora)
SEEDS=(1993 1996 1999)
NGPU=2

echo "==== round2_slurm_grid start: ${#BUDGETS[@]} budgets x ${#METHODS[@]} methods x ${#SEEDS[@]} seeds, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=budget $3=method $4=seed
  local gpu="$1" budget="$2" method="$3" seed="$4"
  local tag="${method}_${budget}mb_s${seed}"
  local config="exps/round2_slurm_grid/${tag}.json"
  local prefix="round2_slurm_grid_omni30t_${budget}mb_${method}_s${seed}"
  # matches trainer.py::_run_bounded_memory's exact output path
  local json="run_logs/boundedmem_${method}_${prefix}_s${seed}.json"
  # matches models/bounded_memory_mixin.py's MetricsLogger tag/out_path exactly
  # (metrics-logging fix, commit 2645b71): a run that completed BEFORE that fix
  # landed has a "done" accuracy json but no (or an incomplete) metrics json --
  # skip-if-done must require BOTH, or a resubmission would silently skip every
  # run that already finished under the old accuracy-only code, permanently
  # leaving it without runtime/persistent-memory/FLOP data.
  local metrics_json="run_logs/final/${method}/metrics_${method}_${prefix}_s${seed}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
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
  echo "==== BUDGET ${BUDGET}MB start $(date) ===="

  # Build the flat 15-job queue: seed-major, method-minor -- seed 1993's 5
  # methods (in order), then seed 1996's 5, then seed 1999's 5.
  JOBS_FIFO="run_logs/round2_slurm_grid/.jobs_${BUDGET}mb.fifo"
  rm -f "$JOBS_FIFO"
  mkfifo "$JOBS_FIFO"

  (
    for SEED in "${SEEDS[@]}"; do
      for METHOD in "${METHODS[@]}"; do
        echo "${METHOD}|${SEED}"
      done
    done
  ) > "$JOBS_FIFO" &
  FIFO_WRITER_PID=$!

  worker() {
    local gpu="$1"
    local line method seed
    # each `read` on the shared fifo atomically consumes exactly one line,
    # so two workers reading the same fd naturally implement a shared work
    # queue with no explicit locking -- standard bash job-queue idiom.
    while IFS='|' read -r method seed; do
      run_one "$gpu" "$BUDGET" "$method" "$seed"
    done < "$JOBS_FIFO"
  }

  for (( g=0; g<NGPU; g++ )); do
    worker "$g" &
  done
  wait
  wait "$FIFO_WRITER_PID" 2>/dev/null || true
  rm -f "$JOBS_FIFO"

  echo "==== BUDGET ${BUDGET}MB done (all methods, all seeds) $(date) ===="
done
echo "==== ALL RUNS COMPLETE (or wall reached) $(date) ===="
