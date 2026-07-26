#!/bin/bash
while kill -0 3720413 2>/dev/null; do sleep 5; done
source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
cd /home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
for tag in seqlora_imagenetr sketchlora_imagenetr_eps005 sketchlora_imagenetr_eps003 sketchlora_imagenetr_eps001 sketchlora_cifar224_eps005 sketchlora_cifar224_eps003 sketchlora_cifar224_eps001; do
  echo "=== starting ${tag} ==="
  python main.py --config "exps/review/seqlora_sketchlora_check/${tag}.json" \
    > "run_logs/stream_smoke/seqlora_sketchlora_check_${tag}.out" 2>&1
  echo "=== finished ${tag} (exit $?) ==="
done
