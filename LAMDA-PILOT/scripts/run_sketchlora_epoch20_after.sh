#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

WAIT_PID="$1"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 5; done

python main.py --config exps/review/task_incremental_imr5t/sketchlora_epoch20.json > run_logs/review_imr5t/sketchlora_epoch20.out 2>&1
