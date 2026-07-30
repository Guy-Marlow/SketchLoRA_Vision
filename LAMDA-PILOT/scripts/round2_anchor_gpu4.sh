#!/bin/bash
# GPU4 lane of the 50MB/15-task anchor grid (user request 2026-07-27).
# sketchlora, then inflora.
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
QUEUE_LOG="run_logs/round2_anchor/queue.log"

for method in sketchlora inflora; do
  patched="exps/round2_anchor/_patched_${method}_50mb_15t_gpu4.json"
  sed "s/PLACEHOLDER/4/" "exps/round2_anchor/${method}_50mb_15t.json" > "$patched"
  echo "[queue] START ${method} [gpu 4] $(date)" >> "$QUEUE_LOG"
  python3 main.py --config "$patched" > "run_logs/round2_anchor/${method}_50mb_15t_gpu4.log" 2>&1
  echo "[queue] DONE  ${method} [gpu 4] $(date)" >> "$QUEUE_LOG"
done
echo "[queue] GPU4 lane ALL DONE $(date)" >> "$QUEUE_LOG"
