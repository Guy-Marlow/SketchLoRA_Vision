#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_inflora ==="
python main.py --config "exps/review/budget250mb_check/cifar224_inflora.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_inflora.out" 2>&1
echo "=== finished cifar224_inflora (exit $?) ==="
echo "=== starting cifar224_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/cifar224_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_rainbowprompt.out" 2>&1
echo "=== finished cifar224_rainbowprompt (exit $?) ==="
echo "=== starting cifar224_treelora ==="
python main.py --config "exps/review/budget250mb_check/cifar224_treelora.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_treelora.out" 2>&1
echo "=== finished cifar224_treelora (exit $?) ==="
echo "=== starting food101_progprompt ==="
python main.py --config "exps/review/budget250mb_check/food101_progprompt.json" > "run_logs/stream_smoke/budget250mb_check_food101_progprompt.out" 2>&1
echo "=== finished food101_progprompt (exit $?) ==="
echo "=== starting food101_sketchlora ==="
python main.py --config "exps/review/budget250mb_check/food101_sketchlora.json" > "run_logs/stream_smoke/budget250mb_check_food101_sketchlora.out" 2>&1
echo "=== finished food101_sketchlora (exit $?) ==="
echo "=== starting imagenetr_olora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_olora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_olora.out" 2>&1
echo "=== finished imagenetr_olora (exit $?) ==="
echo "=== starting imagenetr_seqlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_seqlora.out" 2>&1
echo "=== finished imagenetr_seqlora (exit $?) ==="
echo "=== starting sun397_inflora ==="
python main.py --config "exps/review/budget250mb_check/sun397_inflora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_inflora.out" 2>&1
echo "=== finished sun397_inflora (exit $?) ==="
echo "=== starting sun397_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/sun397_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_sun397_rainbowprompt.out" 2>&1
echo "=== finished sun397_rainbowprompt (exit $?) ==="
echo "=== starting sun397_treelora ==="
python main.py --config "exps/review/budget250mb_check/sun397_treelora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_treelora.out" 2>&1
echo "=== finished sun397_treelora (exit $?) ==="
