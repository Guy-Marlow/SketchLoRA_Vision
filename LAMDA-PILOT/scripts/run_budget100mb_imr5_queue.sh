#!/bin/bash
# Sequential queue for the 8-method budget-mode (100MB/5-chunk) imagenet-r smoke,
# all on GPU0 (only genuinely free GPU -- others hold other users' active jobs).
# Ordered cheapest-epoch-count first for faster incremental feedback; hidelora
# (tuned_epoch=50/chunk) goes last since it dominates runtime.
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

for m in progprompt seqlora olora inflora sketchlora treelora rainbowprompt hidelora; do
    echo "=== starting $m ===" >> run_logs/review_imr5t/queue_budget100mb.log
    python main.py --config exps/review/budget100mb_imr5/${m}.json > run_logs/review_imr5t/budget100mb_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/review_imr5t/queue_budget100mb.log
done
