#!/bin/bash
# Final headline vision experiments: 9 methods x 4 datasets x 3 seeds (108 runs),
# full 20-task splits, batch=48/lr=3e-4, final_metrics enabled for every run.
#
# Usage:
#   bash scripts/run_final_vision_experiments.sh [--data_root PATH] [--gpus 0,1,2,4]
#
# --data_root : where dataset loaders read from (default ./data). Passed through
#               to scripts/data_prep.py, which is idempotent -- already-present
#               datasets are skipped, safe to re-run.
# --gpus      : comma-separated GPU indices to use. If omitted, auto-detects
#               every GPU with ZERO active compute processes at launch time
#               (never shares a GPU with another user's job -- re-checked right
#               before launch, matching this project's standing safety rule).
#               Pass explicitly to override (e.g. if you know a GPU is reserved
#               for you but nvidia-smi shows noise from a stale process).
#
# Each selected GPU gets a sequential queue of however many of the 108 configs
# are assigned to it (round-robin) -- runs one at a time per GPU, GPUs run in
# parallel. Logs: run_logs/final_vision/<tag>.out (stdout/stderr) and
# run_logs/final/<model_name>/metrics_<tag>.json (the actual metrics record,
# written incrementally by utils/metrics_logger.py -- see that file + utils/
# flops.py for exactly what's captured and how).

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (LAMDA-PILOT/)

DATA_ROOT="./data"
GPU_LIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

source /home/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate treelora
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "=== [1/3] dataset setup (idempotent -- skips anything already present) ==="
for ds in cifar100 imagenetr sun397 food101; do
  python scripts/data_prep.py --dataset "$ds" --data_root "$DATA_ROOT"
done

echo "=== [2/3] generating configs (9 methods x 4 datasets x 3 seeds = 108) ==="
python scripts/gen_final_vision_configs.py

echo "=== [3/3] launching across GPUs ==="
mkdir -p run_logs/final_vision

if [[ -z "$GPU_LIST" ]]; then
  # Auto-detect: every GPU index with zero rows in nvidia-smi's compute-apps
  # query. Never picks a GPU another process (yours or anyone else's) is
  # already using -- if that's too conservative for a dedicated allocation,
  # pass --gpus explicitly.
  ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader)
  BUSY_GPUS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  BUSY_INDICES=""
  for uuid in $BUSY_GPUS; do
    idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$uuid" | cut -d',' -f1 | tr -d ' ')
    BUSY_INDICES="$BUSY_INDICES $idx"
  done
  FREE_GPUS=()
  for idx in $ALL_GPUS; do
    if ! echo "$BUSY_INDICES" | grep -qw "$idx"; then
      FREE_GPUS+=("$idx")
    fi
  done
  GPU_LIST=$(IFS=,; echo "${FREE_GPUS[*]}")
  echo "auto-detected idle GPUs: $GPU_LIST"
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
if [[ $N_GPUS -eq 0 ]]; then
  echo "no idle GPUs found (or --gpus produced an empty list) -- aborting, nothing launched."
  exit 1
fi

# collect all generated config tags, then round-robin across N_GPUS lanes
mapfile -t ALL_TAGS < <(ls exps/final_vision/*.json | xargs -n1 basename | sed 's/\.json$//')
echo "${#ALL_TAGS[@]} total configs across ${N_GPUS} GPU(s): ${GPUS[*]}"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  lane_file="run_logs/final_vision/lane_gpu${gpu}.txt"
  > "$lane_file"
  for ((j=i; j<${#ALL_TAGS[@]}; j+=N_GPUS)); do
    echo "${ALL_TAGS[$j]}" >> "$lane_file"
  done
  n_lane=$(wc -l < "$lane_file")
  echo "  GPU $gpu: $n_lane runs"

  # pin device in each of this lane's configs
  while read -r tag; do
    python - "$tag" "$gpu" <<'EOF'
import json, sys
tag, gpu = sys.argv[1], sys.argv[2]
path = f"exps/final_vision/{tag}.json"
d = json.load(open(path))
d["device"] = [gpu]
json.dump(d, open(path, "w"), indent=2)
EOF
  done < "$lane_file"

  nohup bash -c "
    while read -r tag; do
      echo \"=== starting \$tag ===\" >> run_logs/final_vision/queue_gpu${gpu}.log
      python main.py --config exps/final_vision/\${tag}.json > run_logs/final_vision/\${tag}.out 2>&1
      echo \"=== finished \$tag (exit \$?) ===\" >> run_logs/final_vision/queue_gpu${gpu}.log
    done < '$lane_file'
  " > /dev/null 2>&1 &
  disown
done

echo "launched. Watch progress: tail -f run_logs/final_vision/queue_gpu*.log"
echo "Metrics land in: run_logs/final/<model_name>/metrics_final_vision_<dataset>_<method>_s<seed>.json"
