#!/bin/bash
# GPU2 lane of the 50MB/15-task anchor grid (user request 2026-07-27).
# seqlora is already running (PID 556779, started directly, not by this
# script) -- wait for it, then olora, then treelora.
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
QUEUE_LOG="run_logs/round2_anchor/queue.log"

while ps -p 556779 > /dev/null 2>&1; do sleep 10; done
echo "[queue] DONE  seqlora [gpu 2] $(date)" >> "$QUEUE_LOG"

for method in olora treelora; do
  patched="exps/round2_anchor/_patched_${method}_50mb_15t_gpu2.json"
  sed "s/PLACEHOLDER/2/" "exps/round2_anchor/${method}_50mb_15t.json" > "$patched"
  echo "[queue] START ${method} [gpu 2] $(date)" >> "$QUEUE_LOG"
  python3 main.py --config "$patched" > "run_logs/round2_anchor/${method}_50mb_15t_gpu2.log" 2>&1
  echo "[queue] DONE  ${method} [gpu 2] $(date)" >> "$QUEUE_LOG"
done
echo "[queue] GPU2 lane ALL DONE $(date)" >> "$QUEUE_LOG"
