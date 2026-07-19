#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in sun397_seqlora sun397_olora sun397_inflora sun397_sketchlora sun397_treelora sun397_rainbowprompt sun397_progprompt sun397_ease sun397_tuna; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_ttc48_gpu4.log
    python main.py --config exps/review/three_task_check_bs48_lr3e5/${m}.json > run_logs/stream_smoke/three_task_check_bs48lr3e5_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_ttc48_gpu4.log
done
