#!/bin/bash
# Boundary-agnostic ImageNet-R 20t at FINE increments: boundary every 0.5k samples
# (k = mean samples/task; boundary_mult=0.5 -> adapter event every 5 global epochs,
# ~40 chunks over 20 tasks; prior sweep used 1.5k). 4 methods x 3 seeds.
# Seed -> GPU: 1993->0, 1994->1, 1995->2. Method order per GPU (user-specified):
# svdlora -> seqlora -> olora -> inflora. O-LoRA/InfLoRA use lora_n_slots=40 (slot/chunk).
# Skips runs whose .out already has the final "[stream] wrote" line.
set -u
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
DRV=run_logs/_inr20t_bm05_driver.log

SEED_OF=(1993 1994 1995)     # index = GPU id
METHODS=(svdlora seqlora olora inflora)

worker() {
  local gpu="$1" seed="${SEED_OF[$1]}"
  for m in "${METHODS[@]}"; do
    local cfg="${m}_inr20t_sample_bm05_s${seed}"
    if grep -q "\[stream\] wrote" "run_logs/${cfg}.out" 2>/dev/null; then
      echo "==== SKIP (done) ${cfg} $(date) ====" >> "$DRV"; continue
    fi
    echo "==== START [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
    CUDA_VISIBLE_DEVICES="$gpu" $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
    echo "==== DONE  [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
  done
  echo "==== [gpu $gpu] seed ${seed} COMPLETE $(date) ====" >> "$DRV"
}

echo "==== bm05 sweep start $(date) ====" >> "$DRV"
for g in 0 1 2; do worker "$g" & done
wait
echo "ALL INR20T BM05 RUNS COMPLETE $(date)" >> "$DRV"
