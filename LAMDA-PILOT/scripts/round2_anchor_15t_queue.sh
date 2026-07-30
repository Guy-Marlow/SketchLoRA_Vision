#!/bin/bash
# Local 2-GPU queue for the 50MB/15-task anchor grid (user request 2026-07-27):
# SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA, single seed 1993, in that
# fixed order, two workers (GPU 2 and GPU 4) pulling from one shared FIFO --
# same shared-work-queue idiom as scripts/round2_slurm_grid.slurm: whichever
# worker frees up first claims the next unclaimed method, so the first two
# methods start immediately (one per GPU) and the remaining three fill in as
# slots open up, preserving the requested order without a rigid 1:1 GPU
# assignment.
set -uo pipefail

VISION=/home/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION"; exit 1; }
mkdir -p run_logs/round2_anchor
export CUDA_DEVICE_ORDER=PCI_BUS_ID

METHODS=(seqlora sketchlora olora inflora treelora)
GPUS=(2 4)

QUEUE_LOG="run_logs/round2_anchor/queue.log"
echo "==== round2_anchor_15t_queue start $(date) ====" >> "$QUEUE_LOG"

JOBS_FIFO="run_logs/round2_anchor/.jobs.fifo"
rm -f "$JOBS_FIFO"
mkfifo "$JOBS_FIFO"

( for m in "${METHODS[@]}"; do echo "$m"; done ) > "$JOBS_FIFO" &
FIFO_WRITER_PID=$!

worker() {
  local gpu="$1"
  local method
  while IFS= read -r method; do
    local config="exps/round2_anchor/${method}_50mb_15t.json"
    local patched="exps/round2_anchor/_patched_${method}_50mb_15t_gpu${gpu}.json"
    sed "s/PLACEHOLDER/${gpu}/" "$config" > "$patched"
    echo "[queue] START ${method} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
    # NOTE: no CUDA_VISIBLE_DEVICES here -- this project's local (non-SLURM)
    # convention is to bake the real physical GPU index directly into the
    # config's "device" field (matching exps/round2_anchor/*.json precedent
    # and the round2_grid queue watchers), relying on CUDA_DEVICE_ORDER=
    # PCI_BUS_ID above for consistent indexing. Wrapping with
    # CUDA_VISIBLE_DEVICES too would double-restrict (device index would need
    # to be "0" under a visibility restriction, not the real GPU id).
    python3 main.py --config "$patched" \
      > "run_logs/round2_anchor/${method}_50mb_15t_gpu${gpu}.log" 2>&1
    echo "[queue] DONE  ${method} [gpu ${gpu}] $(date)" >> "$QUEUE_LOG"
  done < "$JOBS_FIFO"
}

for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
done
wait
wait "$FIFO_WRITER_PID" 2>/dev/null || true
rm -f "$JOBS_FIFO"

echo "==== round2_anchor_15t_queue ALL DONE $(date) ====" >> "$QUEUE_LOG"
