#!/bin/bash
# OmniBenchmark-1k SVDLoRA(eps=0.005) smokes: first 3/20t, 5/50t, 10/100t tasks.
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs
for cfg in svdlora_omni1k_20t_smoke svdlora_omni1k_50t_smoke svdlora_omni1k_100t_smoke; do
  echo "==== START ${cfg} $(date) ====" >> run_logs/_omni1k_smoke_driver.log
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "==== DONE  ${cfg} $(date) ====" >> run_logs/_omni1k_smoke_driver.log
done
echo "ALL OMNI1K SMOKES COMPLETE $(date)" >> run_logs/_omni1k_smoke_driver.log
