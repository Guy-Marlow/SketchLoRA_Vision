#!/bin/bash
# ImageNet-R grid scheduler v2 (user request 2026-07-28): adds SketchLoRA,
# running AFTER TreeLoRA in method order, to the grid the v1 scheduler started.
# v1 was killed (orchestration only -- its 3 in-flight jobs for the CURRENT
# group, 50mb/seed1993 seqlora+olora+inflora, were left running as orphans,
# reparented to init) because its methods=(...) array was already baked into
# a running bash function and couldn't be updated in place. This script:
#   1. Waits on the 3 known-orphaned PIDs from group 1 (does NOT relaunch them).
#   2. Finishes group 1 by running treelora then sketchlora on it.
#   3. Runs the remaining 5 groups (50mb/1996, 100mb/1993, 100mb/1996,
#      200mb/1993, 200mb/1996) with the full 5-method order from scratch.
# Same GPU-budget-over-time policy as v1: all usable GPUs (1, 2, 4, and 0 once
# FORCE_INCREASE_PID exits) for the first 5 hours from the ORIGINAL v1 start
# time (not reset here), then at most 2 after.
set -uo pipefail

VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
mkdir -p run_logs/imagenetr_grid
QUEUE_LOG="run_logs/imagenetr_grid/queue.log"
echo "==== imagenetr_grid_queue_v2 start (resuming group 1, adding sketchlora) $(date) ====" >> "$QUEUE_LOG"

START_TS="${ORIGINAL_START_TS:-$(date +%s)}"   # preserves v1's original 5h window
FIVE_HOURS=$((5 * 3600))
FORCE_INCREASE_PID="${FORCE_INCREASE_PID:-0}"   # unrelated force_increase job on GPU0

current_max_gpus() {
  local elapsed=$(( $(date +%s) - START_TS ))
  if [ "$elapsed" -lt "$FIVE_HOURS" ]; then
    echo 4
  else
    echo 2
  fi
}

available_gpus() {
  local max="$1"
  local pool=(1 2)
  if [ "$max" -ge 3 ]; then
    pool+=(4)
  fi
  if [ "$max" -ge 4 ] && ! ps -p "$FORCE_INCREASE_PID" > /dev/null 2>&1; then
    pool+=(0)
  fi
  echo "${pool[@]:0:$max}"
}

run_job() {
  local gpu="$1" method="$2" budget="$3" seed="$4"
  local tag="${method}_${budget}mb_s${seed}"
  local config="exps/imagenetr_grid/${tag}.json"
  local patched="exps/imagenetr_grid/_patched_${tag}_gpu${gpu}.json"
  if [ ! -f "$config" ]; then
    echo "[imr] SKIP (no config) ${tag} $(date)" >> "$QUEUE_LOG"
    return
  fi
  sed "s/PLACEHOLDER/${gpu}/" "$config" > "$patched"
  echo "[imr] START ${tag} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py --config "$patched" \
    > "run_logs/imagenetr_grid/${tag}_gpu${gpu}.log" 2>&1
  echo "[imr] DONE  ${tag} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
}

run_batch() {
  # Runs $2 (space-separated method list) for ($budget,$seed) in GPU-budget-
  # sized batches, waiting for each batch before starting the next.
  local budget="$1" seed="$2"
  shift 2
  local methods=("$@")
  local idx=0
  while [ "$idx" -lt "${#methods[@]}" ]; do
    local max_gpus
    max_gpus=$(current_max_gpus)
    read -ra pool <<< "$(available_gpus "$max_gpus")"
    if [ "${#pool[@]}" -eq 0 ]; then
      sleep 30
      continue
    fi
    local pids=()
    for gpu in "${pool[@]}"; do
      [ "$idx" -ge "${#methods[@]}" ] && break
      method="${methods[$idx]}"
      idx=$((idx + 1))
      run_job "$gpu" "$method" "$budget" "$seed" &
      pids+=($!)
    done
    wait "${pids[@]}"
  done
}

# ---- finish group 1 (50mb/1993): wait on the 3 orphaned v1 jobs, then
# treelora, then sketchlora, in that order ----
echo "[imr] waiting on group-1 orphaned jobs (seqlora/olora/inflora, 50mb/1993) $(date)" >> "$QUEUE_LOG"
for pid in 819570 819571 819572; do
  while ps -p "$pid" > /dev/null 2>&1; do sleep 15; done
done
echo "[imr] group-1 orphaned jobs all finished $(date)" >> "$QUEUE_LOG"
run_batch 50 1993 treelora sketchlora
echo "[imr] GROUP DONE  budget=50mb seed=1993 $(date)" >> "$QUEUE_LOG"

# ---- remaining 5 groups, full 5-method order ----
METHODS=(seqlora olora inflora treelora sketchlora)
for BUDGET_SEED in "50 1996" "100 1993" "100 1996" "200 1993" "200 1996"; do
  read -r BUDGET SEED <<< "$BUDGET_SEED"
  echo "[imr] GROUP START budget=${BUDGET}mb seed=${SEED} $(date)" >> "$QUEUE_LOG"
  run_batch "$BUDGET" "$SEED" "${METHODS[@]}"
  echo "[imr] GROUP DONE  budget=${BUDGET}mb seed=${SEED} $(date)" >> "$QUEUE_LOG"
done
echo "==== imagenetr_grid_queue_v2 ALL GROUPS DONE $(date) ====" >> "$QUEUE_LOG"
