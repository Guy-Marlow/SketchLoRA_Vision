#!/bin/bash
# GPU4 lane v2 (user request 2026-07-27): insert a lazy_merge SketchLoRA run
# (same 50MB/15-task/seed-1993 anchor point) ahead of inflora, which had just
# auto-started with no checkpoint progress and was killed to make room.
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
QUEUE_LOG="run_logs/round2_anchor/queue.log"

for method in sketchlora_lazymerge inflora; do
  cfg_name="${method}_50mb_15t.json"
  patched="exps/round2_anchor/_patched_${method}_50mb_15t_gpu4.json"
  sed "s/PLACEHOLDER/4/" "exps/round2_anchor/${cfg_name}" > "$patched"
  echo "[queue] START ${method} [gpu 4] $(date)" >> "$QUEUE_LOG"
  python3 main.py --config "$patched" > "run_logs/round2_anchor/${method}_50mb_15t_gpu4.log" 2>&1
  echo "[queue] DONE  ${method} [gpu 4] $(date)" >> "$QUEUE_LOG"
done
echo "[queue] GPU4 lane v2 ALL DONE $(date)" >> "$QUEUE_LOG"
