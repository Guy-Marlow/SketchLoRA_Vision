#!/bin/bash
# Matched-hyperparameter CIFAR-100 20-task (init_cls=5/inc=5) comparison: isolate ONLY the
# method (and SVD epsilon), everything else identical:
#   rank 8, alpha 32 (scaling 4x), ALL 12 blocks, lr 3e-4, scenario both (CIL+TIL), seed 1993.
# Order: SVDLoRA(eps0.005) -> O-LoRA -> InfLoRA -> SeqLoRA, then SVD epsilon sensitivity {0.01,0.015,0.02}.
set -u
export CUDA_VISIBLE_DEVICES=${1:-4}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
mkdir -p run_logs

CONFIGS=(
  "matched_svd0.005"
  "matched_olora"
  "matched_inflora"
  "matched_seqlora"
  "matched_svd0.01"
  "matched_svd0.015"
  "matched_svd0.02"
)
for cfg in "${CONFIGS[@]}"; do
  echo "================ START ${cfg} $(date) ================" >> run_logs/_matched_driver.log
  $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
  echo "================ DONE  ${cfg} $(date) ================" >> run_logs/_matched_driver.log
done
echo "ALL MATCHED CIFAR RUNS COMPLETE $(date)" >> run_logs/_matched_driver.log