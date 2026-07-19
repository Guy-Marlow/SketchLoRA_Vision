#!/bin/bash
# OmniBenchmark-1k smokes, round 2: 100t (10 tasks) FIRST, then 50t (5 tasks).
# (20t smoke already completed: CIL 73.55 / TIL 88.74 / Fgt 21.1 after 3 tasks.)
set -u
export CUDA_VISIBLE_DEVICES=${1:-0}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
for cfg in svdlora_omni1k_100t_smoke svdlora_omni1k_50t_smoke; do
  echo "==== START ${cfg} $(date) ====" >> run_logs/_omni1k_smoke_driver.log
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "==== DONE  ${cfg} $(date) ====" >> run_logs/_omni1k_smoke_driver.log
done
echo "ROUND2 (100t,50t) SMOKES COMPLETE $(date)" >> run_logs/_omni1k_smoke_driver.log
