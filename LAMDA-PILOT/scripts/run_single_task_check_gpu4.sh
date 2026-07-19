#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in food101_progprompt food101_ease food101_tuna sun397_seqlora sun397_olora sun397_inflora sun397_sketchlora sun397_treelora sun397_rainbowprompt sun397_progprompt sun397_ease sun397_tuna; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_stc_gpu4.log
    python main.py --config exps/review/single_task_check/${m}.json > run_logs/stream_smoke/single_task_check_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_stc_gpu4.log
done
