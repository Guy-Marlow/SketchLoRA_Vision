#!/bin/bash
# Retry 100mb_ca_v2_sharedfull after the main ca_v2_sweep_queue.sh finishes its
# remaining items -- the first attempt crashed on a rand_svd (torch.linalg.svd)
# convergence failure inside the ordinary sketch-fold path (unrelated to the
# shared_full CA covariance code), possibly a transient cusolver numerical
# issue. Waits for the main queue's PID to exit (still GPU2-only, still
# strictly sequential -- never runs concurrently with the main queue).
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
QUEUE_LOG="run_logs/sketchlora_boltons/ca_v2_sweep_queue.log"

MAIN_PID="${1:?usage: ca_v2_sharedfull_retry.sh <main_queue_pid>}"
while ps -p "$MAIN_PID" > /dev/null 2>&1; do sleep 30; done

echo "[ca_v2_retry] main queue finished, retrying sharedfull $(date)" >> "$QUEUE_LOG"
cfg="exps/sketchlora_boltons/100mb_ca_v2_sharedfull.json"
patched="exps/sketchlora_boltons/_patched_100mb_ca_v2_sharedfull_retry_gpu2.json"
sed "s/PLACEHOLDER/2/" "$cfg" > "$patched"
echo "[ca_v2] START 100mb_ca_v2_sharedfull_retry [gpu 2] $(date)" >> "$QUEUE_LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py --config "$patched" \
  > run_logs/sketchlora_boltons/100mb_ca_v2_sharedfull_retry_gpu2.log 2>&1
echo "[ca_v2] DONE  100mb_ca_v2_sharedfull_retry [gpu 2] $(date)" >> "$QUEUE_LOG"
