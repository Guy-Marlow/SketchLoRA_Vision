#!/bin/bash
while kill -0 1462128 2>/dev/null; do sleep 5; done
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_ease ==="
python main.py --config "exps/review/fifty_task_check/cifar224_ease.json" > "run_logs/stream_smoke/fifty_task_check_cifar224_ease.out" 2>&1
echo "=== finished cifar224_ease (exit $?) ==="
echo "=== starting cifar224_rainbowprompt ==="
python main.py --config "exps/review/fifty_task_check/cifar224_rainbowprompt.json" > "run_logs/stream_smoke/fifty_task_check_cifar224_rainbowprompt.out" 2>&1
echo "=== finished cifar224_rainbowprompt (exit $?) ==="
echo "=== starting cifar224_tuna ==="
python main.py --config "exps/review/fifty_task_check/cifar224_tuna.json" > "run_logs/stream_smoke/fifty_task_check_cifar224_tuna.out" 2>&1
echo "=== finished cifar224_tuna (exit $?) ==="
echo "=== starting food101_olora ==="
python main.py --config "exps/review/fifty_task_check/food101_olora.json" > "run_logs/stream_smoke/fifty_task_check_food101_olora.out" 2>&1
echo "=== finished food101_olora (exit $?) ==="
echo "=== starting food101_sketchlora ==="
python main.py --config "exps/review/fifty_task_check/food101_sketchlora.json" > "run_logs/stream_smoke/fifty_task_check_food101_sketchlora.out" 2>&1
echo "=== finished food101_sketchlora (exit $?) ==="
echo "=== starting imagenetr_ease ==="
python main.py --config "exps/review/fifty_task_check/imagenetr_ease.json" > "run_logs/stream_smoke/fifty_task_check_imagenetr_ease.out" 2>&1
echo "=== finished imagenetr_ease (exit $?) ==="
echo "=== starting imagenetr_rainbowprompt ==="
python main.py --config "exps/review/fifty_task_check/imagenetr_rainbowprompt.json" > "run_logs/stream_smoke/fifty_task_check_imagenetr_rainbowprompt.out" 2>&1
echo "=== finished imagenetr_rainbowprompt (exit $?) ==="
echo "=== starting imagenetr_tuna ==="
python main.py --config "exps/review/fifty_task_check/imagenetr_tuna.json" > "run_logs/stream_smoke/fifty_task_check_imagenetr_tuna.out" 2>&1
echo "=== finished imagenetr_tuna (exit $?) ==="
echo "=== starting sun397_olora ==="
python main.py --config "exps/review/fifty_task_check/sun397_olora.json" > "run_logs/stream_smoke/fifty_task_check_sun397_olora.out" 2>&1
echo "=== finished sun397_olora (exit $?) ==="
echo "=== starting sun397_sketchlora ==="
python main.py --config "exps/review/fifty_task_check/sun397_sketchlora.json" > "run_logs/stream_smoke/fifty_task_check_sun397_sketchlora.out" 2>&1
echo "=== finished sun397_sketchlora (exit $?) ==="
