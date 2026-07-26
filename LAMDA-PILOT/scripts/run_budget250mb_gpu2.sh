#!/bin/bash
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting cifar224_progprompt ==="
python main.py --config "exps/review/budget250mb_check/cifar224_progprompt.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_progprompt.out" 2>&1
echo "=== finished cifar224_progprompt (exit $?) ==="
echo "=== starting cifar224_treelora ==="
python main.py --config "exps/review/budget250mb_check/cifar224_treelora.json" > "run_logs/stream_smoke/budget250mb_check_cifar224_treelora.out" 2>&1
echo "=== finished cifar224_treelora (exit $?) ==="
echo "=== starting food101_rainbowprompt ==="
python main.py --config "exps/review/budget250mb_check/food101_rainbowprompt.json" > "run_logs/stream_smoke/budget250mb_check_food101_rainbowprompt.out" 2>&1
echo "=== finished food101_rainbowprompt (exit $?) ==="
echo "=== starting imagenetr_inflora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_inflora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_inflora.out" 2>&1
echo "=== finished imagenetr_inflora (exit $?) ==="
echo "=== starting imagenetr_seqlora ==="
python main.py --config "exps/review/budget250mb_check/imagenetr_seqlora.json" > "run_logs/stream_smoke/budget250mb_check_imagenetr_seqlora.out" 2>&1
echo "=== finished imagenetr_seqlora (exit $?) ==="
echo "=== starting sun397_olora ==="
python main.py --config "exps/review/budget250mb_check/sun397_olora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_olora.out" 2>&1
echo "=== finished sun397_olora (exit $?) ==="
echo "=== starting sun397_sketchlora ==="
python main.py --config "exps/review/budget250mb_check/sun397_sketchlora.json" > "run_logs/stream_smoke/budget250mb_check_sun397_sketchlora.out" 2>&1
echo "=== finished sun397_sketchlora (exit $?) ==="
