#!/bin/bash
#SBATCH --job-name=lorasub_sketch_foldfirst_wave2
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/lorasub_sketch_foldfirst_wave2/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/lorasub_sketch_foldfirst_wave2/slurm_%j.err
#
# LoRA-Subtraction-Sketch (fold-before-eval variant) STANDALONE re-run
# (2026-11-14 user request) -- a direct rip of scripts/
# cllora_lorasub_wave1.slurm, trimmed to ONE method only:
# model_name="lorasub_sketch_foldfirst" (models/lorasub_sketch_foldfirst.py)
# -- the ALREADY-EXISTING, ALREADY-CORRECT fold-before-eval codepath from
# the original scripts/sketch_foldfirst_wave1.slurm campaign. This is NOT
# the same model as cllora_lorasub_wave1's own "lorasub_sketch" sub-
# campaign (model_name="lorasub_sketch", the separate, still-current-timing
# base method) -- the two are distinct files/model_names on the cluster.
# No model code changes are needed for this script; it only orchestrates
# an existing, already-correct method under new output paths.
#
# 5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
# SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) = 15 runs, this
# method only. CL-LoRA-Sketch-FoldFirst, LoRA-Subtraction, and
# cllora_lorasub_wave1's own LoRA-Subtraction-Sketch sub-campoin are NOT
# part of this script -- they continue under cllora_lorasub_wave1.slurm
# as before, independently.
#
# NEW OUTPUT LOCATION ONLY, vs. the original sketch_foldfirst_wave1
# campaign that first established this method's configs -- exps/
# lorasub_sketch_foldfirst_wave2 and run_logs/lorasub_sketch_foldfirst_wave2
# (this script's own SBATCH --output/--error above), so this re-run cannot
# land in or overwrite cllora_lorasub_wave1's in-progress run_logs, nor
# sketch_foldfirst_wave1's original 45-run bundle.
#
# Config generator: scripts/gen_lorasub_sketch_foldfirst_wave2_configs.py
# (run at the top of this script, same as every other campaign here).
# HYPERPARAMETERS are imported byte-identical from the already-settled
# scripts/gen_sketch_foldfirst_wave1_configs.py::lorasub_sketch_foldfirst_cfg.
# `prefix`, however, is OVERRIDDEN to a fresh lorasub_sketch_foldfirst_
# wave2-specific tag (not left as that source campaign's own tag) -- the
# svd_sketching_vision_backup snapshot has no run_logs/final/
# lorasub_sketch_foldfirst/ directory, but that snapshot's run_logs/ tree
# is empty for every model, not just this one, so its absence isn't on
# its own strong enough evidence to justify reusing the original tag the
# way cllora_sketch_foldfirst_cfg's prefix was left unchanged in
# cllora_lorasub_wave1.slurm (which relies on a provable guarantee: every
# prior attempt under ITS tag crashed before writing a metrics file). A
# fresh prefix here sidesteps the question entirely -- even if scripts/
# sketch_foldfirst_wave1.slurm (the ORIGINAL campaign this method's
# config was first built for) is ever resubmitted independently, it
# cannot collide with this standalone re-run's own metrics tag.
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# lorasub_sketch_foldfirst/metrics_<tag>.json's own status=="done") --
# ordinary oracle-mode runs, final_metrics=true throughout.
#
# 30h wall time (lower than cllora_lorasub_wave1's 96h, per instruction --
# this is 1 method x 5 datasets x 3 seeds = 15 runs, not 45). Based on
# this method's own measured per-seed timing under the base (non-
# foldfirst) lorasub_sketch codepath, which this variant only reorders a
# single method call within (no added compute per step): cifar224 ~45min,
# sun397 ~28min, imagenetr ~29min, food101 ~74min, omnibenchmark1k
# ~186min -- 15 runs total ~18h, comfortably inside 30h with margin for
# cluster variance.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every
# other campaign script in this project (this cluster has exactly one GPU
# available).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's
# log -- run_one() below backs up any existing outfile first (same
# convention as every other campaign script here).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=lorasub_sketch_foldfirst_wave2
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
# same idiom as every other campaign script in this project.
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
python scripts/gen_lorasub_sketch_foldfirst_wave2_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. cifar224
  local outdir="${OUT_DIR}/${subdir}"
  mkdir -p "$outdir"
  local outfile="${outdir}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${subdir}/${tag} $(date) ===="
    return
  fi
  if [ -f "$outfile" ]; then
    mv "$outfile" "${outfile}.prev-${SLURM_JOB_ID:-unknown}"
  fi
  echo "==== START ${subdir}/${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${subdir}/${tag} $(date) ===="
}

echo "==== ${CAMPAIGN} start: 15 runs, single GPU $(date) ===="

for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
  while IFS= read -r config; do
    run_one "$config"
  done < <(find "$CONFIG_DIR/$dataset" -name "*.json" 2>/dev/null | sort)
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
