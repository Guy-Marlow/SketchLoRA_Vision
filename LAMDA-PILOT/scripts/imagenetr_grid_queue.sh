#!/bin/bash
# ImageNet-R local grid scheduler (user request 2026-07-28): SeqLoRA, O-LoRA,
# InfLoRA, TreeLoRA x {50,100,200}MB x seeds {1993,1996}, full 20-task split.
#
# Scheduling (explicit user spec, mirrors the H200 cluster grid's budget-major/
# seed-major convention but with a STRICT barrier the cluster script doesn't
# have): budget-major outer loop, seed-major inner loop; for a given (budget,
# seed) GROUP, all 4 methods must FINISH (not just start) before the next
# (budget, seed) group begins. Within a group, methods run in parallel across
# whatever GPU budget currently applies, in BATCHES of that size (so a group
# needing more parallelism than currently allowed splits into sequential
# batches, each fully awaited before the next).
#
# GPU budget changes over real elapsed time (user spec): unrestricted (up to
# all 4 usable GPUs: 1, 2, 4, and 0 once it frees up) for the first 5 hours
# from this script's start, then AT MOST 2 GPUs for everything after. Already-
# running jobs are never killed when the budget drops -- the cap only applies
# to NEW job starts from that point on.
#
# GPU0 is excluded from the pool until the pre-existing guaranteed_admission
# k=5 SketchLoRA run (unrelated work, still being tracked separately) exits --
# this script never touches that job, only waits for its PID to be gone.
# GPU3 (the small ~4GB DGX display card) is never used.
set -uo pipefail

VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
mkdir -p run_logs/imagenetr_grid
QUEUE_LOG="run_logs/imagenetr_grid/queue.log"
echo "==== imagenetr_grid_queue start $(date) ====" >> "$QUEUE_LOG"

START_TS=$(date +%s)
FIVE_HOURS=$((5 * 3600))
GUARANTEED_PID="${GUARANTEED_PID:-0}"   # PID of the unrelated guaranteed_admission run on GPU0

current_max_gpus() {
  local elapsed=$(( $(date +%s) - START_TS ))
  if [ "$elapsed" -lt "$FIVE_HOURS" ]; then
    echo 4
  else
    echo 2
  fi
}

available_gpus() {
  # Prints up to $1 GPU ids, space-separated. 1 and 2 are always eligible; 4
  # eligible for max>=3; 0 eligible for max>=4 AND only once GUARANTEED_PID
  # has exited.
  local max="$1"
  local pool=(1 2)
  if [ "$max" -ge 3 ]; then
    pool+=(4)
  fi
  if [ "$max" -ge 4 ] && ! ps -p "$GUARANTEED_PID" > /dev/null 2>&1; then
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

run_group() {
  local budget="$1" seed="$2"
  local methods=(seqlora olora inflora treelora)
  local idx=0
  echo "[imr] GROUP START budget=${budget}mb seed=${seed} $(date)" >> "$QUEUE_LOG"
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
  echo "[imr] GROUP DONE  budget=${budget}mb seed=${seed} $(date)" >> "$QUEUE_LOG"
}

for BUDGET in 50 100 200; do
  for SEED in 1993 1996; do
    run_group "$BUDGET" "$SEED"
  done
done
echo "==== imagenetr_grid_queue ALL GROUPS DONE $(date) ====" >> "$QUEUE_LOG"
