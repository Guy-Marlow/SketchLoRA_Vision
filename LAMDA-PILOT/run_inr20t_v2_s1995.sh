#!/bin/bash
# ImageNet-R 20-task -- SEED 1995 duplicate of run_inr20t_v2.sh (which runs seed 1993).
# Same 8 configs (4 methods x task/sample), rank 8 / alpha 32 (scaling 4), SVDLoRA eps=0.01.
# Result JSONs/logs are seed-tagged by the harness so they never collide with the seed-1993 run.
set -u
export CUDA_VISIBLE_DEVICES=${1:-2}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs

CONFIGS=(
  "seqlora_inr20t_task_s1995"
  "svdlora_inr20t_task_s1995"
  "olora_inr20t_task_s1995"
  "inflora_inr20t_task_s1995"
)

for cfg in "${CONFIGS[@]}"; do
  echo "================ START ${cfg} $(date) ================" >> run_logs/_inr20t_driver_s1995.log
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "================ DONE  ${cfg} $(date) ================" >> run_logs/_inr20t_driver_s1995.log
done
echo "ALL IMAGENET-R 20-task SEED-1994 RUNS COMPLETE $(date)" >> run_logs/_inr20t_driver_s1995.log