#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_inflora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_inflora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_inflora.out" 2>&1
echo "=== finished cifar224_inflora (exit $?) ==="
echo "=== starting cifar224_seqlora ==="
python main.py --config "exps/review/twenty_task_check/cifar224_seqlora.json" > "run_logs/stream_smoke/twenty_task_check_cifar224_seqlora.out" 2>&1
echo "=== finished cifar224_seqlora (exit $?) ==="
echo "=== starting food101_cllora ==="
python main.py --config "exps/review/twenty_task_check/food101_cllora.json" > "run_logs/stream_smoke/twenty_task_check_food101_cllora.out" 2>&1
echo "=== finished food101_cllora (exit $?) ==="
echo "=== starting food101_progprompt ==="
python main.py --config "exps/review/twenty_task_check/food101_progprompt.json" > "run_logs/stream_smoke/twenty_task_check_food101_progprompt.out" 2>&1
echo "=== finished food101_progprompt (exit $?) ==="
echo "=== starting food101_treelora ==="
python main.py --config "exps/review/twenty_task_check/food101_treelora.json" > "run_logs/stream_smoke/twenty_task_check_food101_treelora.out" 2>&1
echo "=== finished food101_treelora (exit $?) ==="
echo "=== starting imagenetr_inflora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_inflora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_inflora.out" 2>&1
echo "=== finished imagenetr_inflora (exit $?) ==="
echo "=== starting imagenetr_seqlora ==="
python main.py --config "exps/review/twenty_task_check/imagenetr_seqlora.json" > "run_logs/stream_smoke/twenty_task_check_imagenetr_seqlora.out" 2>&1
echo "=== finished imagenetr_seqlora (exit $?) ==="
echo "=== starting sun397_cllora ==="
python main.py --config "exps/review/twenty_task_check/sun397_cllora.json" > "run_logs/stream_smoke/twenty_task_check_sun397_cllora.out" 2>&1
echo "=== finished sun397_cllora (exit $?) ==="
echo "=== starting sun397_progprompt ==="
python main.py --config "exps/review/twenty_task_check/sun397_progprompt.json" > "run_logs/stream_smoke/twenty_task_check_sun397_progprompt.out" 2>&1
echo "=== finished sun397_progprompt (exit $?) ==="
echo "=== starting sun397_treelora ==="
python main.py --config "exps/review/twenty_task_check/sun397_treelora.json" > "run_logs/stream_smoke/twenty_task_check_sun397_treelora.out" 2>&1
echo "=== finished sun397_treelora (exit $?) ==="
