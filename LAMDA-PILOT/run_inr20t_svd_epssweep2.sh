#!/bin/bash
# ImageNet-R 20-task SVDLoRA epsilon-sensitivity sweep (task boundary).
#   CIFAR-parity config: 12 blocks, rank8/alpha32 (x4), lr 3e-4, seed {1993,1994,1995}.
#   New eps values {0.005, 0.0075, 0.0125} to bracket the running eps=0.01 -> 4-point curve.
#   9 runs = 3 eps x 3 seeds. GPU g runs seed's 3 eps sequentially (g0=1993,g1=1994,g2=1995).
#   WAITS for each GPU to free (< 2 GB) before starting, so it auto-launches when the current
#   eps=0.01 runs finish. Skips a run whose .out already has a final Forgetting line.
set -u
PY=/home/gmar762/anaconda3/envs/treelora/bin/python
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p run_logs
DRV=run_logs/_inr20t_epssweep2_driver.log

SEED_OF=(1993 1994 1995)         # index = GPU id
EPS_TAGS=(0150 0175 0200)

wait_free() {   # $1 = gpu id; block until < 2GB used
  while :; do
    u=$(nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    [ -n "$u" ] && [ "$u" -lt 2000 ] && return
    sleep 60
  done
}

worker() {
  local gpu="$1" seed="${SEED_OF[$1]}"
  wait_free "$gpu"
  echo "==== [gpu $gpu] claimed for seed $seed $(date) ====" >> "$DRV"
  for tag in "${EPS_TAGS[@]}"; do
    local cfg="svdlora_inr20t_task_eps${tag}_s${seed}"
    if grep -q "Forgetting (CNN)" "run_logs/${cfg}.out" 2>/dev/null; then
      echo "==== SKIP (done) ${cfg} $(date) ====" >> "$DRV"; continue
    fi
    echo "==== START [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
    CUDA_VISIBLE_DEVICES="$gpu" $PY main.py --config "./exps/${cfg}.json" > "run_logs/${cfg}.out" 2>&1
    echo "==== DONE  [gpu $gpu] ${cfg} $(date) ====" >> "$DRV"
  done
}

echo "==== epssweep start (waiting for GPUs 0/1/2 to free) $(date) ====" >> "$DRV"
for g in 0 1 2; do worker "$g" & done
wait
echo "ALL IMAGENET-R SVD EPS-SWEEP2 RUNS COMPLETE $(date)" >> "$DRV"