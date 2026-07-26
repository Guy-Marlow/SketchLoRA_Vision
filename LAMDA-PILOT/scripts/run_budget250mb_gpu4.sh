#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/cifar224_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_rainbowprompt.out" 2>&1
echo "=== finished cifar224_rainbowprompt (exit $?) ==="
echo "=== starting food101_inflora ==="
python main.py --config "exps/review/budget250mb_check/food101_inflora.json" > "run_logs/stream_smoke/budget250mb_check_food101_inflora.out" 2>&1
echo "=== finished food101_inflora (exit $?) ==="
echo "=== starting food101_seqlora ==="
python main.py --config "exps/review/budget250mb_check/food101_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_food101_seqlora.out" 2>&1
echo "=== finished food101_seqlora (exit $?) ==="
echo "=== starting imagenetr_olora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_olora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_olora.out" 2>&1
echo "=== finished imagenetr_olora (exit $?) ==="
echo "=== starting imagenetr_sketchlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_sketchlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_sketchlora.out" 2>&1
echo "=== finished imagenetr_sketchlora (exit $?) ==="
echo "=== starting sun397_progprompt ==="
python main.py --config "exps/review/budget250mb_check/sun397_progprompt.json" > "run_logs/stream_smoke/budget250mb_check_sun397_progprompt.out" 2>&1
echo "=== finished sun397_progprompt (exit $?) ==="
echo "=== starting sun397_treelora ==="
python main.py --config "exps/review/budget250mb_check/sun397_treelora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_treelora.out" 2>&1
echo "=== finished sun397_treelora (exit $?) ==="
