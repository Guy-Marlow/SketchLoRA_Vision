#!/bin/bash
# LR refinement: svdlora 6e-4 (near its 5e-4 peak), seqlora 1e-3 (above its rising 8e-4).
# 100t pair first, then 50t pair. Appends to the SAME sweep driver log (monitor picks it up).
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
DRV=run_logs/_omni1k_lrsweep_driver.log
for cfg in svdlora_omni1k_100t_smoke_lr6e4 seqlora_omni1k_100t_smoke_lr1e3 \
           svdlora_omni1k_50t_smoke_lr6e4 seqlora_omni1k_50t_smoke_lr1e3; do
  if grep -q "Forgetting (CNN)" "run_logs/${cfg}.out" 2>/dev/null; then
    echo "==== SKIP (done) ${cfg} $(date) ====" >> "$DRV"; continue
  fi
  echo "==== START ${cfg} $(date) ====" >> "$DRV"
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "==== DONE  ${cfg} $(date) ====" >> "$DRV"
done
echo "REFINEMENT RUNS COMPLETE $(date)" >> "$DRV"
