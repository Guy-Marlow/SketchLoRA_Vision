#!/bin/bash
# FLAT-LR (no annealing) counterpart to run_sample_agnostic.sh -- de-confound follow-up.
#   IDENTICAL to the cosine sample sweep except lr_anneal=false (constant init_lr, no cosine
#   schedule at all). Compare stream_*_{inr20t,cifar20t}_sample_flat_s* vs the cosine
#   stream_*_{inr20t,cifar20t}_sample_s* to isolate the schedule effect within streaming
#   (same boundary mode, only the LR schedule toggled).
#   4 methods x 2 datasets x 3 seeds = 24 runs. Seed->GPU: 1993->0, 1994->2, 1995->4.
#   wait_free blocks until each GPU frees (<2GB), so this auto-starts AFTER the cosine
#   sweep's per-GPU quartets finish. Skips a run whose .out already shows "[stream] wrote".
set -u
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p run_logs
DRV=run_logs/_sample_agnostic_flat_driver.log

GPUS=(0 2 4)
SEED_OF=(1993 1994 1995)
METHODS=(seqlora svdlora olora inflora)

wait_free() {
  while :; do
    u=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$u" ] && [ "$u" -lt 2000 ] && return
    sleep 60
  done
}

cfg_name() {   # $1=method $2=dataset(inr|cifar) $3=seed
  local m="$1" d="$2" s="$3"
  if [ "$d" = "inr" ]; then
    if [ "$s" = "1993" ]; then echo "${m}_inr20t_sample_flat"; else echo "${m}_inr20t_sample_flat_s${s}"; fi
  else
    echo "${m}_cifar20t_sample_flat_s${s}"
  fi
}

worker() {
  local gpu="$1" seed="$2"
  wait_free "$gpu"
  echo "==== [gpu $gpu] claimed for seed $seed $(date) ====" >> "$DRV"
  for d in inr cifar; do
    for m in "${METHODS[@]}"; do
      local cfg; cfg=$(cfg_name "$m" "$d" "$seed")
      if [ ! -f "exps/${cfg}.json" ]; then
        echo "==== MISSING config exps/${cfg}.json -- skip $(date) ====" >> "$DRV"; continue
      fi
      if grep -q "\[stream\] wrote" "run_logs/${cfg}.out" 2>/dev/null; then
        echo "==== SKIP (done) ${cfg} $(date) ====" >> "$DRV"; continue
      fi
      echo "==== START [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
      CUDA_VISIBLE_DEVICES="$gpu" $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
      echo "==== DONE  [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
    done
  done
  echo "==== [gpu $gpu] seed $seed COMPLETE $(date) ====" >> "$DRV"
}

echo "==== FLAT-LR sample sweep start (waiting for GPUs to free) $(date) ====" >> "$DRV"
for i in 0 1 2; do worker "${GPUS[$i]}" "${SEED_OF[$i]}" & done
wait
echo "ALL FLAT-LR BOUNDARY-AGNOSTIC RUNS COMPLETE $(date)" >> "$DRV"