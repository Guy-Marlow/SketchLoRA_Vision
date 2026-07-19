#!/bin/bash
# Boundary-AGNOSTIC (sample-boundary / streaming) vision runs on ImageNet-R + CIFAR-100 20t.
#   4 methods {seqlora, svdlora(sketchlora, eps0.01), olora(lamda_1=0.5), inflora}
#   x 2 datasets {inr20t, cifar20t} x 3 seeds {1993,1994,1995} = 24 runs.
#   CIFAR-parity config for ALL: 12 blocks (n_lora_blocks absent), rank8/alpha32 (x4),
#   lr 3e-4, tuned_epoch 10, scenario both. O-LoRA/InfLoRA merge=true. Streaming =
#   adapter event every boundary_mult(1.5)*epochs global epochs, learner unaware of task
#   ends; eval after each fold on completed tasks (models/stream_mixin.py).
#   Seed -> GPU: 1993->0, 1994->2, 1995->4 (40GB A100s; GPU1 busy, GPU3 is the 4GB card).
#   Each GPU runs its 8 configs (inr quartet then cifar quartet) sequentially.
#   WAITS for its GPU to free (<2GB) before starting. Skips a run whose .out already
#   shows the final "[stream] wrote" line.
set -u
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p run_logs
DRV=run_logs/_sample_agnostic_driver.log

GPUS=(0 2 4)                      # index -> gpu id
SEED_OF=(1993 1994 1995)          # index -> seed
METHODS=(seqlora svdlora olora inflora)

wait_free() {   # $1 = gpu id; block until < 2GB used
  while :; do
    u=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$u" ] && [ "$u" -lt 2000 ] && return
    sleep 60
  done
}

cfg_name() {   # $1=method $2=dataset(inr|cifar) $3=seed  ->  echoes config basename
  local m="$1" d="$2" s="$3"
  if [ "$d" = "inr" ]; then
    if [ "$s" = "1993" ]; then echo "${m}_inr20t_sample"; else echo "${m}_inr20t_sample_s${s}"; fi
  else
    echo "${m}_cifar20t_sample_s${s}"
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

echo "==== sample-agnostic sweep start $(date) ====" >> "$DRV"
for i in 0 1 2; do worker "${GPUS[$i]}" "${SEED_OF[$i]}" & done
wait
echo "ALL BOUNDARY-AGNOSTIC VISION RUNS COMPLETE $(date)" >> "$DRV"