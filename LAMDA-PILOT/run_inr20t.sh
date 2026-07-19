#!/bin/bash
# ImageNet-R 20-task (init_cls=10, increment=10 -> 200 classes / 20 tasks) comparison.
# 4 methods: SVDLoRA (adaptive eps=0.005), SeqLoRA, O-LoRA, InfLoRA. lr=5e-3, seed 1993,
# LAMDA-PILOT splits (shuffle=true, scenario both -> CIL+TIL). Sequential on one GPU.
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs

CONFIGS=(
  "sketchlora_inr20t_adapt"
  "seqlora_inr20t"
  "olora_inr20t"
  "inflora_inr20t"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================ START ${cfg} $(date) ================"
  $PY main.py --config "./exps/${cfg}.json" 2>&1 | tee "run_logs/${cfg}.out"
  echo "================ DONE  ${cfg} $(date) ================"
done
echo "ALL IMAGENET-R 20-task RUNS COMPLETE $(date)"
