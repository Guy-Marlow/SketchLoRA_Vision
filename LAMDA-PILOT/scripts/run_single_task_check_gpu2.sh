#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in cifar224_seqlora cifar224_olora cifar224_inflora cifar224_sketchlora cifar224_treelora cifar224_rainbowprompt cifar224_progprompt cifar224_ease cifar224_tuna imagenetr_seqlora imagenetr_olora imagenetr_inflora; do
    echo "=== starting $m ===" >> run_logs/stream_smoke/queue_stc_gpu2.log
    python main.py --config exps/review/single_task_check/${m}.json > run_logs/stream_smoke/single_task_check_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/stream_smoke/queue_stc_gpu2.log
done
