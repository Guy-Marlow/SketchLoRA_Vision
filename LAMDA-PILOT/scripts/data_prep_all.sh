#!/bin/bash
# Dataset setup for the final vision experiments: CIFAR-100, ImageNet-R, SUN397,
# Food101 (OmniBenchmark-1k intentionally excluded -- dropped from the final
# evaluation scope, 2026-07-19). Idempotent -- scripts/data_prep.py skips
# anything already present, safe to re-run.
#
# Usage: bash scripts/data_prep_all.sh [DATA_ROOT]   (default ./data)
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${1:-./data}"
for ds in cifar100 imagenetr sun397 food101; do
  python scripts/data_prep.py --dataset "$ds" --data_root "$DATA_ROOT"
done
