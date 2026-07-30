#!/bin/bash
# 2-GPU-max queue for the SketchLoRA bolt-on factorial (impl_plan_7.27.2026,
# user-narrowed scope 2026-07-27). Uses flock-based atomic job claiming (NOT
# the shared-FIFO idiom used elsewhere in this project -- that pattern has a
# confirmed race: whichever worker's `read` first pairs with the FIFO writer
# drains it, and a slower second worker can block forever in the kernel's
# wait_for_partner state with no writer left. flock avoids this entirely: each
# worker atomically pops one line from a plain job-list file under a lock.)
#
# Worker A claims GPU0 once the golden test (already running there) finishes.
# Worker B claims whichever of GPU2/GPU4 frees first from the pre-existing
# round2_anchor grid (not started or stopped by this script -- just waited on).
set -uo pipefail

VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

QUEUE_LOG="run_logs/sketchlora_boltons/queue.log"
JOBS_FILE="run_logs/sketchlora_boltons/jobs.txt"
LOCKFILE="run_logs/sketchlora_boltons/.jobs.lock"
mkdir -p run_logs/sketchlora_boltons

# All 11 runnable jobs (50mb_off is skipped -- reuses the existing round2_anchor
# result, see scripts/gen_sketchlora_boltons_configs.py). One line per job.
cat > "$JOBS_FILE" <<'EOF'
50mb_fd
50mb_lmplateau
50mb_ca
50mb_fd_lm
50mb_fd_lm_ca
100mb_off
100mb_fd
100mb_lmplateau
100mb_ca
100mb_fd_lm
100mb_fd_lm_ca
EOF
echo "==== sketchlora_boltons_queue start $(date) ====" >> "$QUEUE_LOG"

claim_next_job() {
  # Atomically pop the first line of $JOBS_FILE; echoes it (empty if none left).
  (
    flock -x 200
    job=$(head -n1 "$JOBS_FILE" 2>/dev/null)
    if [ -n "$job" ]; then
      tail -n +2 "$JOBS_FILE" > "${JOBS_FILE}.tmp" && mv "${JOBS_FILE}.tmp" "$JOBS_FILE"
    fi
    echo "$job"
  ) 200>"$LOCKFILE"
}

run_job() {
  local gpu="$1" job="$2"
  local config="exps/sketchlora_boltons/${job}.json"
  local patched="exps/sketchlora_boltons/_patched_${job}_gpu${gpu}.json"
  if [ ! -f "$config" ]; then
    echo "[boltons] SKIP (no config) ${job} $(date)" >> "$QUEUE_LOG"
    return
  fi
  sed "s/PLACEHOLDER/${gpu}/" "$config" > "$patched"
  echo "[boltons] START ${job} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
  python3 main.py --config "$patched" \
    > "run_logs/sketchlora_boltons/${job}_gpu${gpu}.log" 2>&1
  echo "[boltons] DONE  ${job} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
}

worker() {
  local gpu="$1"
  while true; do
    job=$(claim_next_job)
    [ -z "$job" ] && break
    run_job "$gpu" "$job"
  done
  echo "[boltons] worker gpu${gpu} exiting (queue empty) $(date)" >> "$QUEUE_LOG"
}

wait_for_pid() {
  while ps -p "$1" > /dev/null 2>&1; do sleep 10; done
}

# -- Worker A: GPU0, waits for the golden test to finish --
(
  wait_for_pid "$GOLDEN_TEST_PID"
  echo "[boltons] gpu0 worker starting (golden test finished) $(date)" >> "$QUEUE_LOG"
  worker 0
) &

# -- Worker B: whichever of GPU2/GPU4's pre-existing anchor-grid lane finishes
# first (polled, not started/stopped by this script) --
(
  while ps -p "$GPU2_LANE_PID" > /dev/null 2>&1 && ps -p "$GPU4_LANE_PID" > /dev/null 2>&1; do
    sleep 10
  done
  if ! ps -p "$GPU2_LANE_PID" > /dev/null 2>&1; then
    freed_gpu=2
  else
    freed_gpu=4
  fi
  echo "[boltons] gpu${freed_gpu} worker starting (anchor-grid lane finished) $(date)" >> "$QUEUE_LOG"
  worker "$freed_gpu"
) &

wait
echo "==== sketchlora_boltons_queue ALL DONE $(date) ====" >> "$QUEUE_LOG"
