#!/bin/bash
# One-shot cluster setup for the final vision CL experiment grid.
#
# Run this ONCE on a node with internet access (e.g. the login node) before
# submitting scripts/final_vision.slurm -- SLURM compute nodes on many
# clusters have no outbound internet, so both the datasets and the timm
# pretrained-backbone weights need to land in the right place ahead of time.
# Everything this script does is idempotent (safe to re-run).
#
# What it does:
#   1. Prefetches the two timm ViT-B/16 checkpoints every final-grid method
#      needs (vit_base_patch16_224, vit_base_patch16_224_in21k) into the
#      standard torch hub cache (~/.cache/torch/hub/checkpoints), which is
#      where timm==0.6.7 looks for them at train time.
#   2. Downloads/prepares the 4 datasets used by the final grid (CIFAR-100,
#      ImageNet-R, SUN397, Food101) via scripts/data_prep_all.sh.
#
# Usage: bash scripts/setup_cluster.sh [DATA_ROOT]   (default ./data)
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/2] prefetching timm pretrained ViT-B/16 checkpoints ==="
python -c "
import timm
for name in ['vit_base_patch16_224', 'vit_base_patch16_224_in21k']:
    print(f'fetching {name} ...')
    timm.create_model(name, pretrained=True, num_classes=0)
print('done.')
"

echo "=== [2/2] preparing datasets ==="
bash scripts/data_prep_all.sh "${1:-./data}"

echo "=== setup complete ==="
