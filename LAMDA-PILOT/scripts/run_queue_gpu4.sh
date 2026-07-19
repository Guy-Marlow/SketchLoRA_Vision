#!/bin/bash
# Waits for the currently-running olora job (PID passed as $1), then runs the rest
# of GPU4's queue sequentially: sketchlora, treelora, progprompt, cllora.
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

OLORA_PID="$1"
while kill -0 "$OLORA_PID" 2>/dev/null; do sleep 5; done

for m in sketchlora treelora progprompt cllora; do
    echo "=== starting $m ===" >> run_logs/review_imr5t/queue_gpu4.log
    python main.py --config exps/review/task_incremental_imr5t/${m}.json > run_logs/review_imr5t/${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/review_imr5t/queue_gpu4.log
done
