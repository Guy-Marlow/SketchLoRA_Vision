#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
for opt in sgd adamw; do
  echo "=== starting ${opt} ==="
  python main.py --config "exps/review/cllora_hp_check/${opt}.json" \
    > "run_logs/stream_smoke/cllora_hp_check_${opt}.out" 2>&1
  echo "=== finished ${opt} (exit $?) ==="
done
