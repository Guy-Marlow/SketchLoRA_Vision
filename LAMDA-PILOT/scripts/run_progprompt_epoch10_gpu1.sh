#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for d in imagenetr sun397 cifar224; do
    echo "=== starting $d ===" >> run_logs/stream_smoke/queue_progprompt_epoch10.log
    python main.py --config exps/review/progprompt_epoch10_check/${d}.json > run_logs/stream_smoke/progprompt_epoch10_check_${d}.out 2>&1
    echo "=== finished $d (exit $?) ===" >> run_logs/stream_smoke/queue_progprompt_epoch10.log
done
