#!/bin/bash
# Plan B §B2's 20-epoch confirmation re-sweep, across 3 free local A100s.
# InfLoRA gets its own dedicated GPU (flip-watch priority: 1.5e-3 vs 5e-4 at 20ep);
# O-LoRA gets its own GPU; SeqLoRA+TreeLoRA share the third, sequentially.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$(dirname "$0")/.."
mkdir -p run_logs/b2_sweep_20ep

declare -A CENTER=( [seqlora]=0.001 [olora]=0.001 [inflora]=0.0005 [treelora]=0.001 )

run_method () {
  local method=$1 gpu=$2
  local center=${CENTER[$method]}
  local lo hi
  lo=$(python3 -c "print($center/3)")
  hi=$(python3 -c "print($center*3)")
  for dataset in cifar224 imagenetr; do
    for lr in $lo $center $hi; do
      echo "=== $method $dataset lr=$lr (gpu $gpu) ==="
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. python3 scripts/run_lr_sweep_b2.py \
        --method "$method" --dataset "$dataset" --lr "$lr" --device 0 \
        > run_logs/b2_sweep_20ep/log_${method}_${dataset}_lr${lr}.out 2>&1
      echo "=== finished $method $dataset lr=$lr (exit $?) ==="
    done
  done
}

( run_method inflora 1 ) > run_logs/b2_sweep_20ep/queue_inflora.log 2>&1 &
( run_method olora 2 )   > run_logs/b2_sweep_20ep/queue_olora.log 2>&1 &
( run_method seqlora 4; run_method treelora 4 ) > run_logs/b2_sweep_20ep/queue_seqlora_treelora.log 2>&1 &
wait
echo "ALL B2 20-EPOCH SWEEP RUNS COMPLETE"
