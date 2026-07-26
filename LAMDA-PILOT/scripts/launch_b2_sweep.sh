#!/bin/bash
# Launches Plan B §B2's 3-point lr sweep across the 4 free local A100s.
# One method per GPU; each GPU runs its method's 2 datasets x 3 lr points (6 runs) sequentially.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$(dirname "$0")/.."
mkdir -p run_logs/b2_sweep

declare -A CENTER=( [seqlora]=0.001 [olora]=0.001 [inflora]=0.0005 [treelora]=0.001 )
declare -A GPU=( [seqlora]=0 [olora]=1 [inflora]=2 [treelora]=4 )

for method in seqlora olora inflora treelora; do
  gpu=${GPU[$method]}
  center=${CENTER[$method]}
  lo=$(python3 -c "print($center/3)")
  hi=$(python3 -c "print($center*3)")
  (
    for dataset in cifar224 imagenetr; do
      for lr in $lo $center $hi; do
        echo "=== $method $dataset lr=$lr (gpu $gpu) ==="
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. python3 scripts/run_lr_sweep_b2.py \
          --method "$method" --dataset "$dataset" --lr "$lr" --device 0 \
          > run_logs/b2_sweep/log_${method}_${dataset}_lr${lr}.out 2>&1
        echo "=== finished $method $dataset lr=$lr (exit $?) ==="
      done
    done
  ) > run_logs/b2_sweep/queue_${method}.log 2>&1 &
done
wait
echo "ALL B2 SWEEP RUNS COMPLETE"
