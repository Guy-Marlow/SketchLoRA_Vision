#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in imagenetr_seqlora imagenetr_olora imagenetr_inflora imagenetr_sketchlora imagenetr_treelora imagenetr_rainbowprompt imagenetr_progprompt imagenetr_ease imagenetr_tuna; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_ttc48b_gpu1.log
    python main.py --config exps/review/three_task_check_bs48_lr3e4/${m}.json > run_logs/stream_smoke/three_task_check_bs48lr3e4_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_ttc48b_gpu1.log
done
