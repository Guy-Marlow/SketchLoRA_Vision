#!/bin/bash
# impl_plan_7.28.2026 sec 2 CA repair sweep -- SEQUENTIAL on GPU2 ONLY, per the
# user's thermal-driven single-GPU-for-5-hours constraint (2026-07-28 18:19
# NZST -> ~23:19 NZST). Control (f, plain CA) is NOT re-run here -- it already
# exists as exps/sketchlora_boltons/100mb_ca.json's completed result
# (top1=70.99, top5=89.34) and is reused as the comparison baseline. This
# queue runs the 5 v2 variants that ARE new: logit-adjust-only (d),
# shared_full and low_rank_diag covariance (b), real-feature mixing at 50% (c),
# and a ca_steps=100 + early-stop arm (a). The combined arm (e, a-winner x
# b-winner) is NOT included -- it is correctly sequenced AFTER this sweep's
# own results are known, per the plan's own ordering.
set -uo pipefail
VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || exit 1
mkdir -p run_logs/sketchlora_boltons
QUEUE_LOG="run_logs/sketchlora_boltons/ca_v2_sweep_queue.log"
echo "==== ca_v2_sweep_queue start (GPU2 only) $(date) ====" >> "$QUEUE_LOG"

TAGS=(ca_v2_logitadjust ca_v2_sharedfull ca_v2_lowrankdiag ca_v2_realmix50 ca_v2_steps100_earlystop)

for tag in "${TAGS[@]}"; do
  cfg="exps/sketchlora_boltons/100mb_${tag}.json"
  patched="exps/sketchlora_boltons/_patched_100mb_${tag}_gpu2.json"
  sed "s/PLACEHOLDER/2/" "$cfg" > "$patched"
  echo "[ca_v2] START 100mb_${tag} [gpu 2] $(date)" >> "$QUEUE_LOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID python3 main.py --config "$patched" \
    > "run_logs/sketchlora_boltons/100mb_${tag}_gpu2.log" 2>&1
  echo "[ca_v2] DONE  100mb_${tag} [gpu 2] $(date)" >> "$QUEUE_LOG"
done
echo "==== ca_v2_sweep_queue ALL DONE $(date) ====" >> "$QUEUE_LOG"
