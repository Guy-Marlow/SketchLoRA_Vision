#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
for ds in imagenetr sun397 cifar224; do
  echo "=== starting ${ds} ==="
  python main.py --config "exps/review/progprompt_epoch15_check/${ds}.json" \
    > "run_logs/stream_smoke/progprompt_epoch15_check_${ds}.out" 2>&1
  echo "=== finished ${ds} (exit $?) ==="
done
