#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in sun397_250mb_inflora food101_250mb_seqlora food101_250mb_olora; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_sf_gpu2.log
    python main.py --config exps/review/stream_smoke/${m}.json > run_logs/stream_smoke/${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_sf_gpu2.log
done
