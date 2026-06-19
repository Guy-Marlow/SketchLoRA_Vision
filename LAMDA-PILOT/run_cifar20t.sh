#!/bin/bash
# CIFAR-100 20-task (init_cls=5, increment=5) benchmark — 5 methods, sequential on one GPU.
# Defaults preserved per method (inflora lr 5e-4, rest 3e-4; bs 48). seed 1993.
set -u
export CUDA_VISIBLE_DEVICES=${1:-1}
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs

CONFIGS=(
  "sketchlora_r8_l3_20t"
  "sketchlora_20t"
  "seqlora_20t"
  "olora_20t"
  "inflora_20t"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================ START ${cfg} $(date) ================"
  $PY main.py --config "./exps/${cfg}.json" 2>&1 | tee "run_logs/cifar20t_${cfg}.out"
  echo "================ DONE  ${cfg} $(date) ================"
done
echo "ALL CIFAR-100 20-task RUNS COMPLETE $(date)"
