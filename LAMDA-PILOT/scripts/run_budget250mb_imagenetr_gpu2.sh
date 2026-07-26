#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting imagenetr_seqlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_seqlora.out" 2>&1
echo "=== finished imagenetr_seqlora (exit $?) ==="
echo "=== starting imagenetr_olora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_olora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_olora.out" 2>&1
echo "=== finished imagenetr_olora (exit $?) ==="
echo "=== starting imagenetr_inflora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_inflora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_inflora.out" 2>&1
echo "=== finished imagenetr_inflora (exit $?) ==="
echo "=== starting imagenetr_sketchlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_sketchlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_sketchlora.out" 2>&1
echo "=== finished imagenetr_sketchlora (exit $?) ==="
echo "=== starting imagenetr_treelora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_treelora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_treelora.out" 2>&1
echo "=== finished imagenetr_treelora (exit $?) ==="
echo "=== starting imagenetr_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_rainbowprompt.out" 2>&1
echo "=== finished imagenetr_rainbowprompt (exit $?) ==="
echo "=== starting imagenetr_progprompt ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_progprompt.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_progprompt.out" 2>&1
echo "=== finished imagenetr_progprompt (exit $?) ==="
