#!/bin/bash
# ImageNet-R 20-task (init_cls=10, increment=10 -> 200 classes / 20 tasks) comparison.
#   4 methods x 2 boundary modes = 8 runs. SVDLoRA(sketchlora, adaptive eps=0.005), SeqLoRA,
#   O-LoRA(lamda_1=0.5), InfLoRA(lamb=lame=0.98). lr=5e-3, seed 1993, n_lora_blocks=3,
#   lora_rank 10, tuned_epoch 10, scenario both (CIL+TIL). boundary task = per-task-end;
#   boundary sample = streaming (every 1.5x task instances, learner unaware). Sequential, 1 GPU.
#   Ordered task-quartet first (simpler/cheaper SeqLoRA first to surface any data-load issue
#   fast), then the streaming sample quartet. Skips a run whose result JSON already exists.
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs

CONFIGS=(
  "seqlora_inr20t_task"
  "svdlora_inr20t_task"
  "olora_inr20t_task"
  "inflora_inr20t_task"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================ START ${cfg} $(date) ================" >> run_logs/_inr20t_driver.log
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "================ DONE  ${cfg} $(date) ================" >> run_logs/_inr20t_driver.log
done
echo "ALL IMAGENET-R 20-task RUNS COMPLETE $(date)" >> run_logs/_inr20t_driver.log