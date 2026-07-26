#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_olora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_olora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_olora.out" 2>&1
echo "=== finished cifar224_olora (exit $?) ==="
echo "=== starting cifar224_sketchlora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_sketchlora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_sketchlora.out" 2>&1
echo "=== finished cifar224_sketchlora (exit $?) ==="
echo "=== starting food101_ease ==="
python main.py --config "exps/review/twenty_task_check/food101_ease.json" > "run_logs/stream_smoke/twenty_task_check_food101_ease.out" 2>&1
echo "=== finished food101_ease (exit $?) ==="
echo "=== starting food101_rainbowprompt ==="
python main.py --config "exps/review/twenty_task_check/food101_rainbowprompt.json" > "run_logs/stream_smoke/twenty_task_check_food101_rainbowprompt.out" 2>&1
echo "=== finished food101_rainbowprompt (exit $?) ==="
echo "=== starting food101_tuna ==="
python main.py --config "exps/review/twenty_task_check/food101_tuna.json" > "run_logs/stream_smoke/twenty_task_check_food101_tuna.out" 2>&1
echo "=== finished food101_tuna (exit $?) ==="
echo "=== starting imagenetr_olora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_olora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_olora.out" 2>&1
echo "=== finished imagenetr_olora (exit $?) ==="
echo "=== starting imagenetr_sketchlora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_sketchlora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_sketchlora.out" 2>&1
echo "=== finished imagenetr_sketchlora (exit $?) ==="
echo "=== starting sun397_ease ==="
python main.py --config "exps/review/twenty_task_check/sun397_ease.json" > "run_logs/stream_smoke/twenty_task_check_sun397_ease.out" 2>&1
echo "=== finished sun397_ease (exit $?) ==="
echo "=== starting sun397_rainbowprompt ==="
python main.py --config "exps/review/twenty_task_check/sun397_rainbowprompt.json" > "run_logs/stream_smoke/twenty_task_check_sun397_rainbowprompt.out" 2>&1
echo "=== finished sun397_rainbowprompt (exit $?) ==="
echo "=== starting sun397_tuna ==="
python main.py --config "exps/review/twenty_task_check/sun397_tuna.json" > "run_logs/stream_smoke/twenty_task_check_sun397_tuna.out" 2>&1
echo "=== finished sun397_tuna (exit $?) ==="
