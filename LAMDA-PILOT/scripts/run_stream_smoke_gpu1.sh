#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in progprompt seqlora rainbowprompt; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_gpu1.log
    python main.py --config exps/review/stream_smoke/cifar224_${m}.json > run_logs/stream_smoke/cifar224_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_gpu1.log
done
