#!/bin/bash
# force_increase (k=1), rank_cap disabled, sequential on GPU0 -- 50MB then 100MB
# (user request 2026-07-28: the capped version risked the at-cap eviction branch
# overriding the growth floor entirely once rank_cap=128 was reached, collapsing
# force_increase into plain bounded_eviction behavior; rank_cap=None removes that
# entirely -- see models/sketchlora.py's _compress: cap = rank_cap if not None
# else composite_rank, so the at-cap branch is structurally unreachable here).
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
mkdir -p run_logs/sketchlora_boltons
QUEUE_LOG="run_logs/sketchlora_boltons/force_increase_nocap_queue.log"
echo "==== force_increase_nocap_queue start $(date) ====" >> "$QUEUE_LOG"

for budget in 50 100; do
  tag="${budget}mb_force_increase_k1_nocap"
  echo "[nocap] START ${tag} [gpu 0] $(date)" >> "$QUEUE_LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py --config "exps/sketchlora_boltons/${tag}.json" \
    > "run_logs/sketchlora_boltons/${tag}_gpu0.log" 2>&1
  echo "[nocap] DONE  ${tag} [gpu 0] $(date)" >> "$QUEUE_LOG"
done
echo "==== force_increase_nocap_queue ALL DONE $(date) ====" >> "$QUEUE_LOG"
