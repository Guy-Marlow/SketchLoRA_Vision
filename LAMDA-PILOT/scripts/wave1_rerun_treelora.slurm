#!/bin/bash
#SBATCH --job-name=wave1_rerun_treelora
#SBATCH --time=28:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/wave1_rerun_treelora/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/wave1_rerun_treelora/slurm_%j.err
#
# Canon re-run of TreeLoRA's wave1_final grid, 2026-08-16 user request
# (originally scoped to O-LoRA + TreeLoRA together; narrowed to TreeLoRA
# only per explicit follow-up in the same request). The ORIGINAL wave1_final
# TreeLoRA runs (2026-08-09) predate commit 23c6405 (2026-08-10), which
# removed TreeLoRA's dense-slot folding and fixed its fold-specific
# persistent_state() override -- a real change to how persistent memory is
# COUNTED (mirrors this session's O-LoRA recompute: the fixed fold-buffer
# term is gone, replaced by summing what's actually still allocated).
# Accuracy should be unaffected in principle (factored vs. folded forward are
# mathematically identical outputs), but nothing has actually re-run the
# current code end-to-end to confirm that -- this campaign produces real,
# current-code numbers for both accuracy and memory, not just a hand-derived
# memory correction layered on stale logs.
#
# SCOPE: all 5 wave1_final datasets (CIFAR-100, Food-101, ImageNet-R,
# OmniBenchmark-1K, SUN397) x 3 seeds (1993/1996/1999) = 15 runs. Every
# config is a VERBATIM copy of its wave1_final source (same task/epoch/LR/
# reg settings) with only `prefix` changed -- see
# scripts/gen_wave1_rerun_treelora_configs.py's own docstring for exactly
# why a distinct prefix is required here (short version: reusing the
# original prefix would make this campaign's completion check see the stale
# pre-fix run's already-"done" metrics JSON and skip the rerun entirely on
# any resumed submission).
#
# WALL TIME (28h budget, single GPU by construction -- --gres=gpu:1,
# CUDA_VISIBLE_DEVICES=0, sequential loop): measured directly off the
# ORIGINAL wave1_final TreeLoRA .out logs' own timestamps (2026-08-09, seed
# 1993) per dataset -- OmniBenchmark-1K is the dominant cost at ~3h23m (100
# tasks vs. 10-20 for the other four); the other four datasets mirror
# O-LoRA's own per-dataset timings closely on this shared LoRA scaffold
# (CIFAR-100 ~48.5min, Food-101 ~81min, ImageNet-R ~32min, SUN397 ~28min,
# confirmed directly for O-LoRA this session, assumed comparable for
# TreeLoRA given both share backbone/vit_lora.py and the same per-dataset
# task/epoch counts). Per-seed total: ~6.5h. x3 seeds =~ 19.5h, leaving
# ~8.5h of headroom under the 28h budget. Resumable end-to-end regardless
# (see run_one below) if that's not enough -- resubmit, already-finished
# runs are skipped.
#
# RESUMABLE BY CONSTRUCTION: run_one() skips any config whose final_metrics
# JSON already shows status=="done", via scripts/_bankcap_run_done.py (fully
# generic despite the name -- it only reads a config's model_name/dataset/
# prefix/seed and reconstructs trainer.py's own metrics-path formula; no
# bank-cap-specific logic, reused verbatim from the bankcap_wave1_
# imagenetr20t campaign rather than duplicated).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=wave1_rerun_treelora
CONFIG_DIR="exps/$CAMPAIGN"
OUT_DIR="run_logs/$CAMPAIGN"
mkdir -p "$CONFIG_DIR" "$OUT_DIR"

LOCKFILE="$OUT_DIR/.$CAMPAIGN.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another $CAMPAIGN.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Pre-flight sanity check -- single-GPU by construction, sequential loop,
# same idiom as every other script in this project.
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
if [ "$NGPU" != "1" ]; then
  echo "NOTE: torch.cuda.device_count()=${NGPU} (expected 1 for this job's --gres=gpu:1) -- proceeding on CUDA_VISIBLE_DEVICES=0 regardless."
fi
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

echo "==== config generation start $(date) ===="
python scripts/gen_wave1_rerun_treelora_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local outfile="${OUT_DIR}/${tag}.out"

  # Completion check: the final_metrics JSON's own status=="done" (every
  # config in this campaign sets final_metrics=true), NOT a grep of the .out
  # log's "Average Accuracy (CNN):" line -- see scripts/_bankcap_run_done.py
  # and bankcap_wave1_imagenetr20t.slurm's matching run_one for the full
  # reasoning (that line prints once per TASK, not once per completed run,
  # so grepping it can falsely mark a wall-time-killed run as done forever).
  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

CONFIG_COUNT=$(ls "$CONFIG_DIR"/*.json 2>/dev/null | wc -l)
echo "==== ${CAMPAIGN} start: ${CONFIG_COUNT} configs (treelora, 5 datasets, 3 seeds each), single GPU $(date) ===="

# Dataset order matches wave1_final's own convention, OmniBenchmark-1K last
# since it's by far the longest-running dataset per seed; seed ascending.
for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
  for cfg in "$CONFIG_DIR/treelora_${dataset}_s"*.json; do
    run_one "$cfg"
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE $(date) ===="
