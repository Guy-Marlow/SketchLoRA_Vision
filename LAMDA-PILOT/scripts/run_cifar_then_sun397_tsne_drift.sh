#!/bin/bash
# Runs the fixed-rank SketchLoRA embed-drift (tSNE) study on CIFAR-100-10t
# FIRST, then SUN397-10t -- both write exact/sketch-boundary embeddings via
# models/sketchlora.py's embed_drift_dir hooks, same convention as the
# ImageNet-R 5t/20t runs already collected under run_logs/tsne_drift/.
#
# CIFAR-100 data: already present locally (./data/cifar-100-python), but this
# still runs scripts/data_prep.py explicitly as a visible "ensure downloaded"
# step rather than relying silently on iCIFAR224's own download=True call
# inside main.py -- a no-op here since torchvision's integrity check will
# just confirm the existing files and skip re-downloading.
#
# SUN397 data: materialized separately (data_prep.py --dataset sun397
# --data_root ~/Documents, already running in the background as of this
# script's creation) into Documents/sun397, junctioned at
# ./data/sun397 -> C:/Users/gmar762/Documents/sun397. This script WAITS on
# that download's own completion marker before touching SUN397 at all, so it
# is safe to launch this script before that download finishes.
#
# Runs sequentially, single GPU, in this exact order (CIFAR fully completes
# -- all 10 tasks + extraction -- before SUN397 starts).
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "FATAL: cannot cd to LAMDA-PILOT root"; exit 1; }

PY="/c/Users/gmar762/AppData/Local/anaconda3/envs/treelora/python.exe"
SUN397_PREP_LOG="/c/Users/gmar762/Documents/sun397_data_prep_log.txt"

echo "==== [1/4] ensure CIFAR-100 present $(date) ===="
"$PY" scripts/data_prep.py --dataset cifar100 --data_root ./data
echo "==== CIFAR-100 present $(date) ===="

echo "==== [2/4] CIFAR-100-10t embed-drift run start $(date) ===="
"$PY" main.py --config exps/tsne_drift/sketchlora_fixedrank_cifar224_10t_s1993.json \
  > run_logs/tsne_drift/train_log_cifar224_10t_s1993.txt 2>&1
echo "==== CIFAR-100-10t embed-drift run done $(date) ===="

echo "==== [3/4] waiting on SUN397 data download to finish $(date) ===="
while true; do
  if [ -f "$SUN397_PREP_LOG" ] && grep -q "^\[sun397\] done ->" "$SUN397_PREP_LOG"; then
    echo "==== SUN397 data confirmed complete $(date) ===="
    break
  fi
  if [ -f "$SUN397_PREP_LOG" ] && grep -qi "Traceback\|Error" "$SUN397_PREP_LOG"; then
    echo "FATAL: sun397 data_prep log shows an error -- check $SUN397_PREP_LOG"
    exit 1
  fi
  echo "  ... sun397 data not marked done yet, waiting 60s ($(date))"
  sleep 60
done

echo "==== [4/4] SUN397-10t embed-drift run start $(date) ===="
"$PY" main.py --config exps/tsne_drift/sketchlora_fixedrank_sun397_10t_s1993.json \
  > run_logs/tsne_drift/train_log_sun397_10t_s1993.txt 2>&1
echo "==== SUN397-10t embed-drift run done $(date) ===="

echo "==== ALL DONE (cifar224-10t then sun397-10t) $(date) ===="
