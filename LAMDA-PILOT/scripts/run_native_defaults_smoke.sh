#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
for tag in inflora_cifar224 tuna_cifar224 tuna_imagenetr ease_cifar224 ease_imagenetr; do
  echo "=== starting ${tag} ==="
  python main.py --config "exps/review/native_defaults_smoke/${tag}.json" \
    > "run_logs/stream_smoke/native_defaults_smoke_${tag}.out" 2>&1
  echo "=== finished ${tag} (exit $?) ==="
done
