#!/bin/bash
# Waits for the GPU0 lane driver (seqlora -> sketchlora -> olora) to finish, then
# continues the SAME GPU0-only sequence with the remaining 5 methods in the
# user-specified order (inflora, treelora, rainbowprompt/progprompt, hidelora last).
# Consolidated to GPU0 only -- GPU1/2/4 all have other researchers' active jobs.
set -uo pipefail
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

DRIVER_PID="$1"
while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 5; done

for m in inflora treelora rainbowprompt progprompt hidelora; do
    echo "=== starting $m ===" >> run_logs/review_imr5t/queue_omni8_gpu0.log
    python main.py --config exps/review/budget250mb_omni8/${m}.json > run_logs/review_imr5t/budget250mb_omni8_${m}.out 2>&1
    echo "=== finished $m (exit $?) ===" >> run_logs/review_imr5t/queue_omni8_gpu0.log
done
