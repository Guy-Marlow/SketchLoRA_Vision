#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in food101_seqlora food101_olora food101_inflora food101_sketchlora food101_treelora food101_rainbowprompt food101_progprompt food101_ease food101_tuna; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_ttc48_gpu2.log
    python main.py --config exps/review/three_task_check_bs48_lr3e5/${m}.json > run_logs/stream_smoke/three_task_check_bs48lr3e5_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_ttc48_gpu2.log
done
