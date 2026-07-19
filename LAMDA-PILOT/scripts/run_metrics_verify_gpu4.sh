#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in olora rainbowprompt progprompt ease tuna; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_metrics_verify.log
    python main.py --config exps/review/metrics_verify/${m}.json > run_logs/stream_smoke/metrics_verify_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_metrics_verify.log
done
