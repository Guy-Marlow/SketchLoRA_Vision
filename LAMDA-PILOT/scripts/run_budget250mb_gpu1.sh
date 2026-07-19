#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_olora ==="
python main.py --config "exps/review/budget250mb_check/cifar224_olora.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_olora.out" 2>&1
echo "=== finished cifar224_olora (exit $?) ==="
echo "=== starting cifar224_seqlora ==="
python main.py --config "exps/review/budget250mb_check/cifar224_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_seqlora.out" 2>&1
echo "=== finished cifar224_seqlora (exit $?) ==="
echo "=== starting food101_inflora ==="
python main.py --config "exps/review/budget250mb_check/food101_inflora.json" > "run_logs/stream_smoke/budget250mb_check_food101_inflora.out" 2>&1
echo "=== finished food101_inflora (exit $?) ==="
echo "=== starting food101_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/food101_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_food101_rainbowprompt.out" 2>&1
echo "=== finished food101_rainbowprompt (exit $?) ==="
echo "=== starting food101_treelora ==="
python main.py --config "exps/review/budget250mb_check/food101_treelora.json" > "run_logs/stream_smoke/budget250mb_check_food101_treelora.out" 2>&1
echo "=== finished food101_treelora (exit $?) ==="
echo "=== starting imagenetr_progprompt ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_progprompt.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_progprompt.out" 2>&1
echo "=== finished imagenetr_progprompt (exit $?) ==="
echo "=== starting imagenetr_sketchlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_sketchlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_sketchlora.out" 2>&1
echo "=== finished imagenetr_sketchlora (exit $?) ==="
echo "=== starting sun397_olora ==="
python main.py --config "exps/review/budget250mb_check/sun397_olora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_olora.out" 2>&1
echo "=== finished sun397_olora (exit $?) ==="
echo "=== starting sun397_seqlora ==="
python main.py --config "exps/review/budget250mb_check/sun397_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_seqlora.out" 2>&1
echo "=== finished sun397_seqlora (exit $?) ==="
