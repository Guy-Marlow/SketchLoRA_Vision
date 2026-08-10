#!/bin/bash
#SBATCH --job-name=ce2_omnibenchmark1k_10t
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce2_omnibenchmark1k_10t/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce2_omnibenchmark1k_10t/slurm_%j.err
#
# CE2 campaign (2026-08-10, user request): all 9 in-scope methods (HiDeLoRA
# excluded) on a genuine 10-task OmniBenchmark-1K split (init_cls=100,
# increment=100 -- 1000 classes / 10 tasks, NOT a stop_after_tasks-truncated
# slice of the 100-task split, so every method's task-count-dependent
# scheduling, e.g. InfLoRA's DualGPM threshold ramp, completes properly
# within these 10 tasks rather than reading as "10% through a 100-task run").
# Single seed (1993) -- this is a compute-characterization campaign, not an
# accuracy campaign; no per-seed variance claim is made.
#
# Uses trainer.py's NEW ce2_enabled path (utils/ce2_profiler.py): exactly 3
# torch.profiler(with_flops=True) sessions per task -- the whole training
# loop, the whole eval, and the method's own boundary op (SketchLoRA's
# _compress, O-LoRA's _refresh_orth_cache, InfLoRA's _init_lora_A +
# _update_dualgpm, TreeLoRA's tree.end_task, CL-LoRA/EASE's replace_fc,
# TUNA's _compute_mean + classifer_align; SeqLoRA/RainbowPrompt have none).
# Deliberately independent of and mutually exclusive with the OLD
# ce_ledger/R2-probe machinery (trainer.py guarantees model._ce_boundary_ctrl
# stays unset whenever ce2_enabled is set) -- no nested-profiler risk.
#
# tuned_epoch=20 / batch_size=48 fixed across all 9 configs (each method's own
# already-established LR/hyperparameters otherwise unchanged from
# wave1_final/cllora_ease_tuna_rainbow_sens) -- the training BUDGET is a
# controlled variable, so any train-time difference between methods reflects
# extra per-step work (e.g. the shared dense-fold's heavier forward/backward,
# O-LoRA's orthogonality penalty), not different amounts of training.
#
# WALL-TIME: 240h is a CONSERVATIVE, largely-unvalidated budget, not a
# measured one. Known unprofiled train_seconds_total figures from the
# wave1_final 100-task run (grouping-invariant to first order -- see
# ce_profiling_methodology memory) top out around InfLoRA's ~14746s (~4.1h)
# for the 5 LoRA-family methods; CL-LoRA/EASE/TUNA/RainbowPrompt have NEVER
# been run to completion on omnibenchmark1k at this task count (only
# stop_after_tasks=10 truncated LR-sweep runs exist), so their real cost here
# is unknown. torch.profiler(with_flops=True) wrapping EVERY step's whole
# training loop (not narrow/sampled, per user's explicit request this session)
# adds real, uncharacterized overhead on top of that. 240h leaves generous
# headroom for both unknowns; the resumable skip-check below means a wall-kill
# loses no completed method's data.
#
# RESUMABLE / LOCKFILE: same convention as wave1_final.slurm -- skips a
# method whose ce2 JSON already has all 10 task records; refuses to start a
# second instance of itself.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/ce2_omnibenchmark1k_10t

LOCKFILE="run_logs/ce2_omnibenchmark1k_10t/.ce2_omnibenchmark1k_10t.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another ce2_omnibenchmark1k_10t.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

PREFLIGHT_ERR="$(mktemp)"
NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>"$PREFLIGHT_ERR")
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat "$PREFLIGHT_ERR"
  rm -f "$PREFLIGHT_ERR"
  exit 1
fi
rm -f "$PREFLIGHT_ERR"
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

# method order matches final_vision.slurm's convention; SeqLoRA first since
# it's the CE baseline every other method gets scored against.
METHODS=(seqlora sketchlora olora inflora treelora cllora rainbowprompt ease tuna)

echo "==== ce2_omnibenchmark1k_10t start: ${#METHODS[@]} methods, single seed (1993), single GPU $(date) ===="

run_one() {
  local method="$1"
  local config="exps/ce2_omnibenchmark1k_10t/${method}_omnibenchmark1k_10t_s1993.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${method} $(date) ===="
    return
  fi

  # Reproduce trainer.py's ce2 tag/path formula exactly (out_dir=run_logs/ce2/
  # <model_name>, filename=ce2_<model_name>_<dataset>_<prefix>_s<seed>.json)
  # rather than hand-deriving it in bash, to avoid drifting from the real
  # formula the way wave1_final.slurm's sketchlora-diag-path derivation does
  # for the same reason.
  local ce2_json
  ce2_json=$(python -c "
import json
c = json.load(open('${config}'))
seed = c['seed'][0] if isinstance(c['seed'], list) else c['seed']
tag = '{}_{}_{}_s{}'.format(c['model_name'], c['dataset'], c.get('prefix') or 'task', seed)
print('run_logs/ce2/{}/ce2_{}.json'.format(c['model_name'], tag))
")

  if [ -f "$ce2_json" ] && python -c "
import json, sys
j = json.load(open('${ce2_json}'))
sys.exit(0 if len(j.get('tasks', [])) >= 10 else 1)
" 2>/dev/null; then
    echo "==== SKIP (done, 10/10 tasks) ${method} $(date) ===="
    return
  fi

  echo "==== START ${method} $(date) ===="
  python main.py --config "$config" \
    > "run_logs/ce2_omnibenchmark1k_10t/${method}.out" 2>&1 || true
  echo "==== DONE  ${method} $(date) ===="
}

for METHOD in "${METHODS[@]}"; do
  run_one "$METHOD"
done

echo "==== ce2_omnibenchmark1k_10t ALL METHODS ATTEMPTED $(date) ===="
echo "==== computing CE-vs-SeqLoRA scores $(date) ===="
python scripts/compute_ce2_scores.py --out-dir run_logs/ce2 \
  --methods "${METHODS[@]}" --dataset omnibenchmark1k --seed 1993 \
  > run_logs/ce2_omnibenchmark1k_10t/ce2_scores.txt 2>&1 || true
echo "==== ce2_omnibenchmark1k_10t DONE $(date) ===="
