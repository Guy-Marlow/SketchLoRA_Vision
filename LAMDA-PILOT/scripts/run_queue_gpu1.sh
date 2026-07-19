#!/bin/bash
# Waits for the currently-running seqlora job (PID passed as $1), then runs the rest
# of GPU1's queue sequentially: inflora, hidelora, rainbowprompt, ease, tuna.
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

SEQLORA_PID="$1"
while kill -0 "$SEQLORA_PID" 2>/dev/null; do sleep 5; done

for m in inflora hidelora rainbowprompt ease tuna; do
    echo "=== starting $m ===" >> run_logs/review_imr5t/queue_gpu1.log
    python main.py --config exps/review/task_incremental_imr5t/${m}.json > run_logs/review_imr5t/${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/review_imr5t/queue_gpu1.log
done
