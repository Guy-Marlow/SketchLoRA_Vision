#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_cllora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_cllora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_cllora.out" 2>&1
echo "=== finished cifar224_cllora (exit $?) ==="
echo "=== starting cifar224_progprompt ==="
python main.py --config "exps/review/twenty_task_check/cifar224_progprompt.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_progprompt.out" 2>&1
echo "=== finished cifar224_progprompt (exit $?) ==="
echo "=== starting cifar224_treelora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_treelora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_treelora.out" 2>&1
echo "=== finished cifar224_treelora (exit $?) ==="
echo "=== starting food101_inflora ==="
python main.py --config "exps/review/twenty_task_check/food101_inflora.json" > "run_logs/stream_smoke/twenty_task_check_food101_inflora.out" 2>&1
echo "=== finished food101_inflora (exit $?) ==="
echo "=== starting food101_seqlora ==="
python main.py --config "exps/review/twenty_task_check/food101_seqlora.json" > "run_logs/stream_smoke/twenty_task_check_food101_seqlora.out" 2>&1
echo "=== finished food101_seqlora (exit $?) ==="
echo "=== starting imagenetr_cllora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_cllora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_cllora.out" 2>&1
echo "=== finished imagenetr_cllora (exit $?) ==="
echo "=== starting imagenetr_progprompt ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_progprompt.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_progprompt.out" 2>&1
echo "=== finished imagenetr_progprompt (exit $?) ==="
echo "=== starting imagenetr_treelora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_treelora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_treelora.out" 2>&1
echo "=== finished imagenetr_treelora (exit $?) ==="
echo "=== starting sun397_inflora ==="
python main.py --config "exps/review/twenty_task_check/sun397_inflora.json" > "run_logs/stream_smoke/twenty_task_check_sun397_inflora.out" 2>&1
echo "=== finished sun397_inflora (exit $?) ==="
echo "=== starting sun397_seqlora ==="
python main.py --config "exps/review/twenty_task_check/sun397_seqlora.json" > "run_logs/stream_smoke/twenty_task_check_sun397_seqlora.out" 2>&1
echo "=== finished sun397_seqlora (exit $?) ==="
