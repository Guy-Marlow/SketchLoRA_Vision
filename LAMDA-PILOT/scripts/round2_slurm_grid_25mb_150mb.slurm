#!/bin/bash
#SBATCH --job-name=round2_slurm_grid_25mb_150mb
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_25mb150mb_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/round2_slurm_grid/slurm_25mb150mb_%j.err
#
# Follow-up to scripts/round2_slurm_grid.slurm /
# scripts/round2_slurm_grid_completion_and_50mb.slurm (2026-08-10 user
# request): adds two new memory-budget points, 25MB and 150MB, to the
# existing SeqLoRA/SketchLoRA/O-LoRA/InfLoRA/TreeLoRA x {25,50,100,150,200}MB
# x 3 seeds (1993/1996/1999) OmniBenchmark-1K bounded-memory grid. Same
# per-method hyperparameters as every other budget point in this series (see
# scripts/gen_round2_slurm_grid_configs.py's METHOD_CFG -- unchanged, just
# two more BUDGETS entries), same harness, same output directory.
#
# NOT OVERWRITING EXISTING DATA: identical run_one() naming convention as the
# original scripts (config/.out/accuracy-json/metrics-json paths are all
# keyed on "{method}_{budget}mb_s{seed}" / "..._{budget}mb_..."), and 25/150
# are budget values that have never been used before in this campaign -- so
# every path this script writes to is, by construction, one that neither
# round2_slurm_grid.slurm's 100/200MB runs nor
# round2_slurm_grid_completion_and_50mb.slurm's 50MB runs ever touched.
# Nothing here can collide with or overwrite existing 50/100/200MB logging;
# it all lands in the same run_logs/round2_slurm_grid/ directory alongside it.
#
# SINGLE GPU, 8 CPUs, 48GB RAM (2026-08-10 user spec) -- unlike the original
# two scripts in this series (2 GPUs, 18 CPUs, 128GB, two-worker shared-FIFO
# queue), this one runs its 30-job list (2 budgets x 5 methods x 3 seeds)
# SEQUENTIALLY on a single GPU. No FIFO/worker-pool machinery needed at
# NGPU=1 -- a plain for-loop over the job list already is the single-worker
# case of that pattern.
#
# WALL TIME (168h = 7 days), derived as instructed from the slowest method's
# own observed 50MB completion time, conservatively, times total run count --
# not a generic guess:
#   Measured directly from run_logs/round2_slurm_grid/*_50mb_s*.out (first ->
#   last timestamped log line per run): InfLoRA is the slowest method at
#   50MB across all 3 seeds (306.0 / 301.4 / 312.6 min; SketchLoRA, O-LoRA,
#   TreeLoRA, SeqLoRA are all faster, 248-287 min mean) -- consistent with
#   the CE investigation earlier this project found InfLoRA's DualGPM
#   subspace update genuinely costs two extra full forward passes per task,
#   not a measurement artifact.
#   Conservative per-run budget: InfLoRA's slowest single observed run
#   (312.6 min) rounded up to 320 min (5h20m), applied uniformly to EVERY
#   run in this script's 30-job list (not just InfLoRA's) -- i.e. assuming,
#   conservatively, that every method takes as long as the slowest method
#   observed anywhere in the reference 50MB data.
#   320 min x 30 runs = 9600 min = 160h -> rounded up to 168h (7 days) for
#   startup/config-gen/skip-check overhead margin.
#   NOTE this is a real risk either way: 168h may exceed this cluster's
#   partition/QOS wall-time cap (check before submitting -- wave1_final.slurm
#   ran successfully at 192h on this same cluster, so multi-day jobs are not
#   inherently rejected here, but confirm rather than assume). The
#   resumable/skip-check design below means a wall-kill + resubmission simply
#   continues from whatever's still incomplete, same as every other script in
#   this series -- it does not need to complete in one submission to be safe
#   to run.
#
# RESUMABLE / SKIP-IF-DONE: identical convention to
# round2_slurm_grid_completion_and_50mb.slurm (accuracy json "partial":false
# AND metrics json "status":"done" both required).
#
# LOCKFILE: refuses to start alongside this script's own prior instance, OR
# alongside the original 2-GPU round2_slurm_grid scripts (all three share the
# same run_logs/round2_slurm_grid/ output tree and GPU 0 could otherwise be
# double-booked).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/round2_slurm_grid

for OTHER_LOCK in \
  "run_logs/round2_slurm_grid/.round2_slurm_grid.lock" \
  "run_logs/round2_slurm_grid/.round2_slurm_grid_completion_and_50mb.lock" \
  "run_logs/round2_slurm_grid/.round2_slurm_grid_treelora_only.lock"; do
  if [ -f "$OTHER_LOCK" ] && kill -0 "$(cat "$OTHER_LOCK")" 2>/dev/null; then
    echo "FATAL: $OTHER_LOCK (PID $(cat "$OTHER_LOCK")) appears to be running -- refusing to start alongside it. Wait for it to finish or delete the lockfile if stale."
    exit 1
  fi
done
LOCKFILE="run_logs/round2_slurm_grid/.round2_slurm_grid_25mb_150mb.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another instance of this script (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Pre-flight sanity check (same convention as the rest of this series).
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/round2_slurm_grid_25mb150mb_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/round2_slurm_grid_25mb150mb_preflight_$$.err
  rm -f /tmp/round2_slurm_grid_25mb150mb_preflight_$$.err
  exit 1
fi
rm -f /tmp/round2_slurm_grid_25mb150mb_preflight_$$.err

echo "==== config generation start $(date) ===="
python scripts/gen_round2_slurm_grid_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1=method $2=budget $3=seed -- identical naming/skip-check convention to
  # round2_slurm_grid.slurm / round2_slurm_grid_completion_and_50mb.slurm's
  # own run_one(), just single-GPU (no $gpu arg -- CUDA_VISIBLE_DEVICES=0 is
  # fixed for the whole script).
  local method="$1" budget="$2" seed="$3"
  local tag="${method}_${budget}mb_s${seed}"
  local config="exps/round2_slurm_grid/${tag}.json"
  local prefix="round2_slurm_grid_omni30t_${budget}mb_${method}_s${seed}"
  local json="run_logs/boundedmem_${method}_${prefix}_s${seed}.json"
  local metrics_json="run_logs/final/${method}/metrics_${method}_${prefix}_s${seed}.json"

  if [ ! -f "$config" ]; then
    echo "==== FATAL (missing config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$json" ] && grep -q '"partial": false' "$json" \
     && [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (already done, accuracy+metrics both complete) ${tag} $(date) ===="
    return
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" > "run_logs/round2_slurm_grid/${tag}.out" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

# Budget-major (all 25MB drains before 150MB starts), seed-major/method-minor
# within a budget -- same ordering convention as the rest of this series,
# just consumed sequentially since there's only one worker here.
JOBS=(
  "seqlora|25|1993" "sketchlora|25|1993" "olora|25|1993" "inflora|25|1993" "treelora|25|1993"
  "seqlora|25|1996" "sketchlora|25|1996" "olora|25|1996" "inflora|25|1996" "treelora|25|1996"
  "seqlora|25|1999" "sketchlora|25|1999" "olora|25|1999" "inflora|25|1999" "treelora|25|1999"
  "seqlora|150|1993" "sketchlora|150|1993" "olora|150|1993" "inflora|150|1993" "treelora|150|1993"
  "seqlora|150|1996" "sketchlora|150|1996" "olora|150|1996" "inflora|150|1996" "treelora|150|1996"
  "seqlora|150|1999" "sketchlora|150|1999" "olora|150|1999" "inflora|150|1999" "treelora|150|1999"
)

echo "==== round2_slurm_grid_25mb_150mb start: ${#JOBS[@]} jobs, single GPU $(date) ===="
for job in "${JOBS[@]}"; do
  IFS='|' read -r method budget seed <<< "$job"
  run_one "$method" "$budget" "$seed"
done
echo "==== ALL 25MB/150MB RUNS COMPLETE (or wall reached) $(date) ===="
