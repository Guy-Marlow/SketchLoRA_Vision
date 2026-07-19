#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in imagenetr_sketchlora imagenetr_treelora imagenetr_rainbowprompt imagenetr_progprompt imagenetr_ease imagenetr_tuna food101_seqlora food101_olora food101_inflora food101_sketchlora food101_treelora food101_rainbowprompt; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_stc_gpu1.log
    python main.py --config exps/review/single_task_check/${m}.json > run_logs/stream_smoke/single_task_check_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_stc_gpu1.log
done
