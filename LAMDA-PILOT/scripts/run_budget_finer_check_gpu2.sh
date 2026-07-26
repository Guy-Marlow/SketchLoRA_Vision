#!/bin/bash
while kill -0 1779402 2>/dev/null; do sleep 5; done
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
echo "=== starting 100mb_cifar224_progprompt ==="
python main.py --config "exps/review/budget_finer_check/100mb_cifar224_progprompt.json" > "run_logs/stream_smoke/budget_finer_check_100mb_cifar224_progprompt.out" 2>&1
echo "=== finished 100mb_cifar224_progprompt (exit $?) ==="
echo "=== starting 100mb_cifar224_treelora ==="
python main.py --config "exps/review/budget_finer_check/100mb_cifar224_treelora.json" > "run_logs/stream_smoke/budget_finer_check_100mb_cifar224_treelora.out" 2>&1
echo "=== finished 100mb_cifar224_treelora (exit $?) ==="
echo "=== starting 100mb_food101_rainbowprompt ==="
python main.py --config "exps/review/budget_finer_check/100mb_food101_rainbowprompt.json" > "run_logs/stream_smoke/budget_finer_check_100mb_food101_rainbowprompt.out" 2>&1
echo "=== finished 100mb_food101_rainbowprompt (exit $?) ==="
echo "=== starting 100mb_imagenetr_inflora ==="
python main.py --config "exps/review/budget_finer_check/100mb_imagenetr_inflora.json" > "run_logs/stream_smoke/budget_finer_check_100mb_imagenetr_inflora.out" 2>&1
echo "=== finished 100mb_imagenetr_inflora (exit $?) ==="
echo "=== starting 100mb_imagenetr_seqlora ==="
python main.py --config "exps/review/budget_finer_check/100mb_imagenetr_seqlora.json" > "run_logs/stream_smoke/budget_finer_check_100mb_imagenetr_seqlora.out" 2>&1
echo "=== finished 100mb_imagenetr_seqlora (exit $?) ==="
echo "=== starting 100mb_sun397_olora ==="
python main.py --config "exps/review/budget_finer_check/100mb_sun397_olora.json" > "run_logs/stream_smoke/budget_finer_check_100mb_sun397_olora.out" 2>&1
echo "=== finished 100mb_sun397_olora (exit $?) ==="
echo "=== starting 100mb_sun397_sketchlora ==="
python main.py --config "exps/review/budget_finer_check/100mb_sun397_sketchlora.json" > "run_logs/stream_smoke/budget_finer_check_100mb_sun397_sketchlora.out" 2>&1
echo "=== finished 100mb_sun397_sketchlora (exit $?) ==="
echo "=== starting 150mb_cifar224_progprompt ==="
python main.py --config "exps/review/budget_finer_check/150mb_cifar224_progprompt.json" > "run_logs/stream_smoke/budget_finer_check_150mb_cifar224_progprompt.out" 2>&1
echo "=== finished 150mb_cifar224_progprompt (exit $?) ==="
echo "=== starting 150mb_cifar224_treelora ==="
python main.py --config "exps/review/budget_finer_check/150mb_cifar224_treelora.json" > "run_logs/stream_smoke/budget_finer_check_150mb_cifar224_treelora.out" 2>&1
echo "=== finished 150mb_cifar224_treelora (exit $?) ==="
echo "=== starting 150mb_food101_rainbowprompt ==="
python main.py --config "exps/review/budget_finer_check/150mb_food101_rainbowprompt.json" > "run_logs/stream_smoke/budget_finer_check_150mb_food101_rainbowprompt.out" 2>&1
echo "=== finished 150mb_food101_rainbowprompt (exit $?) ==="
echo "=== starting 150mb_imagenetr_inflora ==="
python main.py --config "exps/review/budget_finer_check/150mb_imagenetr_inflora.json" > "run_logs/stream_smoke/budget_finer_check_150mb_imagenetr_inflora.out" 2>&1
echo "=== finished 150mb_imagenetr_inflora (exit $?) ==="
echo "=== starting 150mb_imagenetr_seqlora ==="
python main.py --config "exps/review/budget_finer_check/150mb_imagenetr_seqlora.json" > "run_logs/stream_smoke/budget_finer_check_150mb_imagenetr_seqlora.out" 2>&1
echo "=== finished 150mb_imagenetr_seqlora (exit $?) ==="
echo "=== starting 150mb_sun397_olora ==="
python main.py --config "exps/review/budget_finer_check/150mb_sun397_olora.json" > "run_logs/stream_smoke/budget_finer_check_150mb_sun397_olora.out" 2>&1
echo "=== finished 150mb_sun397_olora (exit $?) ==="
echo "=== starting 150mb_sun397_sketchlora ==="
python main.py --config "exps/review/budget_finer_check/150mb_sun397_sketchlora.json" > "run_logs/stream_smoke/budget_finer_check_150mb_sun397_sketchlora.out" 2>&1
echo "=== finished 150mb_sun397_sketchlora (exit $?) ==="
echo "=== starting 200mb_cifar224_progprompt ==="
python main.py --config "exps/review/budget_finer_check/200mb_cifar224_progprompt.json" > "run_logs/stream_smoke/budget_finer_check_200mb_cifar224_progprompt.out" 2>&1
echo "=== finished 200mb_cifar224_progprompt (exit $?) ==="
echo "=== starting 200mb_cifar224_treelora ==="
python main.py --config "exps/review/budget_finer_check/200mb_cifar224_treelora.json" > "run_logs/stream_smoke/budget_finer_check_200mb_cifar224_treelora.out" 2>&1
echo "=== finished 200mb_cifar224_treelora (exit $?) ==="
echo "=== starting 200mb_food101_rainbowprompt ==="
python main.py --config "exps/review/budget_finer_check/200mb_food101_rainbowprompt.json" > "run_logs/stream_smoke/budget_finer_check_200mb_food101_rainbowprompt.out" 2>&1
echo "=== finished 200mb_food101_rainbowprompt (exit $?) ==="
echo "=== starting 200mb_imagenetr_inflora ==="
python main.py --config "exps/review/budget_finer_check/200mb_imagenetr_inflora.json" > "run_logs/stream_smoke/budget_finer_check_200mb_imagenetr_inflora.out" 2>&1
echo "=== finished 200mb_imagenetr_inflora (exit $?) ==="
echo "=== starting 200mb_imagenetr_seqlora ==="
python main.py --config "exps/review/budget_finer_check/200mb_imagenetr_seqlora.json" > "run_logs/stream_smoke/budget_finer_check_200mb_imagenetr_seqlora.out" 2>&1
echo "=== finished 200mb_imagenetr_seqlora (exit $?) ==="
echo "=== starting 200mb_sun397_olora ==="
python main.py --config "exps/review/budget_finer_check/200mb_sun397_olora.json" > "run_logs/stream_smoke/budget_finer_check_200mb_sun397_olora.out" 2>&1
echo "=== finished 200mb_sun397_olora (exit $?) ==="
echo "=== starting 200mb_sun397_sketchlora ==="
python main.py --config "exps/review/budget_finer_check/200mb_sun397_sketchlora.json" > "run_logs/stream_smoke/budget_finer_check_200mb_sun397_sketchlora.out" 2>&1
echo "=== finished 200mb_sun397_sketchlora (exit $?) ==="
echo "=== starting 50mb_cifar224_progprompt ==="
python main.py --config "exps/review/budget_finer_check/50mb_cifar224_progprompt.json" > "run_logs/stream_smoke/budget_finer_check_50mb_cifar224_progprompt.out" 2>&1
echo "=== finished 50mb_cifar224_progprompt (exit $?) ==="
echo "=== starting 50mb_cifar224_treelora ==="
python main.py --config "exps/review/budget_finer_check/50mb_cifar224_treelora.json" > "run_logs/stream_smoke/budget_finer_check_50mb_cifar224_treelora.out" 2>&1
echo "=== finished 50mb_cifar224_treelora (exit $?) ==="
echo "=== starting 50mb_food101_rainbowprompt ==="
python main.py --config "exps/review/budget_finer_check/50mb_food101_rainbowprompt.json" > "run_logs/stream_smoke/budget_finer_check_50mb_food101_rainbowprompt.out" 2>&1
echo "=== finished 50mb_food101_rainbowprompt (exit $?) ==="
echo "=== starting 50mb_imagenetr_inflora ==="
python main.py --config "exps/review/budget_finer_check/50mb_imagenetr_inflora.json" > "run_logs/stream_smoke/budget_finer_check_50mb_imagenetr_inflora.out" 2>&1
echo "=== finished 50mb_imagenetr_inflora (exit $?) ==="
echo "=== starting 50mb_imagenetr_seqlora ==="
python main.py --config "exps/review/budget_finer_check/50mb_imagenetr_seqlora.json" > "run_logs/stream_smoke/budget_finer_check_50mb_imagenetr_seqlora.out" 2>&1
echo "=== finished 50mb_imagenetr_seqlora (exit $?) ==="
echo "=== starting 50mb_sun397_olora ==="
python main.py --config "exps/review/budget_finer_check/50mb_sun397_olora.json" > "run_logs/stream_smoke/budget_finer_check_50mb_sun397_olora.out" 2>&1
echo "=== finished 50mb_sun397_olora (exit $?) ==="
echo "=== starting 50mb_sun397_sketchlora ==="
python main.py --config "exps/review/budget_finer_check/50mb_sun397_sketchlora.json" > "run_logs/stream_smoke/budget_finer_check_50mb_sun397_sketchlora.out" 2>&1
echo "=== finished 50mb_sun397_sketchlora (exit $?) ==="
