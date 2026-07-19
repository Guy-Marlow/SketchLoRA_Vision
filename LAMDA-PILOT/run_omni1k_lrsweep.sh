#!/bin/bash
# OmniBenchmark-1k LR sweep (seed 1993): svdlora(eps.005) vs seqlora,
# lr {8e-4,5e-4,3e-4,1e-4,8e-5}, 10 tasks of the 100t regime + 5 tasks of the 50t regime.
# svdlora@3e-4 cells reuse the completed smoke runs (svdlora_omni1k_{100t,50t}_smoke.out).
# 100t regime FIRST (where LR matters most), lr descending, svd/seq pairs adjacent.
# Skips any run whose .out already has the final Forgetting line.
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
DRV=run_logs/_omni1k_lrsweep_driver.log

CONFIGS=()
for reg in 100t 50t; do
  for tag in 8e4 5e4 3e4 1e4 8e5; do
    [ "$tag" != "3e4" ] && CONFIGS+=("svdlora_omni1k_${reg}_smoke_lr${tag}")
    CONFIGS+=("seqlora_omni1k_${reg}_smoke_lr${tag}")
  done
done

echo "==== lrsweep start: ${#CONFIGS[@]} runs $(date) ====" >> "$DRV"
for cfg in "${CONFIGS[@]}"; do
  if grep -q "Forgetting (CNN)" "run_logs/${cfg}.out" 2>/dev/null; then
    echo "==== SKIP (done) ${cfg} $(date) ====" >> "$DRV"; continue
  fi
  echo "==== START ${cfg} $(date) ====" >> "$DRV"
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "==== DONE  ${cfg} $(date) ====" >> "$DRV"
done
echo "ALL OMNI1K LR-SWEEP RUNS COMPLETE $(date)" >> "$DRV"
