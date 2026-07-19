#!/bin/bash
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
for m in rainbowprompt progprompt hidelora; do
    echo "=== starting $m ===" >> run_logs/review_imr5t/queue_omni8_gpu4.log
    python main.py --config exps/review/budget250mb_omni8/${m}.json > run_logs/review_imr5t/budget250mb_omni8_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/review_imr5t/queue_omni8_gpu4.log
done
