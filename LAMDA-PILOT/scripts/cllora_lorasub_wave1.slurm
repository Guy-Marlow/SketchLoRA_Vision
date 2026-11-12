#!/bin/bash
#SBATCH --job-name=cllora_lorasub_wave1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/cllora_lorasub_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/cllora_lorasub_wave1/slurm_%j.err
#
# CL-LoRA-Sketch-FoldFirst + LoRA-Subtraction + LoRA-Subtraction-Sketch
# campaign (2026-11-12 user request): three sequential sub-campaigns, in
# this order -- (1) CL-LoRA-Sketch-FoldFirst (model_name=cllora_sketch_
# foldfirst, the "v3" design: orth_a_side=true, use_block_weight_
# scaling=false, sketchlora_align_weight=0.75), (2) LoRA-Subtraction
# (model_name=lorasub, growing bank, log_bank_rank=true), (3)
# LoRA-Subtraction-Sketch (model_name=lorasub_sketch, fixed 2-slot sketch,
# no bank-rank logging -- not applicable to a fixed-size structure).
#
# 5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
# SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) x 3 methods = 45 runs.
#
# Every config is byte-identical to its already-settled source campaign
# (not re-derived) -- see scripts/gen_cllora_lorasub_wave1_configs.py's own
# docstring for exactly which generator function each method's configs
# come from, and why `prefix` is deliberately left as each source
# campaign's own tag rather than retagged for this one.
#
# CL-LoRA-Sketch-FoldFirst was the variant that crashed with
# KeyError('use_distillation') before this session's fix (scripts/
# gen_sketch_foldfirst_wave1_configs.py's CLLORA_COMMON was missing
# use_distillation/alpha/beta -- both now added, all 30 affected configs
# across both fold-first campaigns regenerated and verified, commit
# 8800724). LoRA-Subtraction/LoRA-Subtraction-Sketch depend on backbone/
# vit_lorasub.py + models/lorasub.py + models/lorasub_sketch.py's DRS-
# probe/init/prototype fixes, smoke-validated this session (CIFAR-224
# 5-task: clean full-curve pass; ImageNet-R 10-task: 9/10 task evals clean
# before being intentionally stopped) and committed in the same commit.
#
# Config generator: scripts/gen_cllora_lorasub_wave1_configs.py (run at the
# top of this script, same as every other campaign here -- configs are
# regenerated, not hand-committed).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <model_name>/metrics_<tag>.json's own status=="done") -- all three
# sub-campaigns are ordinary oracle-mode runs, final_metrics=true
# throughout. Because `prefix` is left unchanged from each source
# campaign, this resumability check will also recognize (and skip) any
# run from the ORIGINAL sketch_foldfirst_wave1 / bank_rank_wave1 /
# lorasub_sdlora_sketch_wave1 campaigns that already completed under the
# same tag -- intentional, these are genuinely the same logical runs.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every
# other campaign script in this project (this cluster has exactly one GPU
# available).
#
# 96h wall time: 45 full-length runs including OmniBenchmark-1k-100t x3
# methods x3 seeds = 9 of the most expensive runs in this set -- this will
# almost certainly need multiple resubmissions, same as every other long
# campaign here; the RESUMABLE design above is what makes that safe, not a
# sign this is mis-scoped.
#
# A retry's plain `>` would silently overwrite a prior failed attempt's
# log -- run_one() below backs up any existing outfile first (same
# convention as every other campaign script here).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=cllora_lorasub_wave1
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
python scripts/gen_cllora_lorasub_wave1_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. lorasub/cifar224
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

# Run order: cllora_sketch_foldfirst, then lorasub, then lorasub_sketch
# (exactly the order the user asked for) -- dataset-minor within each,
# `sort` keeping seed order deterministic. No special-casing
# OmniBenchmark-1k -- every run in this set is already full-length.
echo "==== ${CAMPAIGN} start: 45 runs, single GPU $(date) ===="

for method in cllora_sketch_foldfirst lorasub lorasub_sketch; do
  for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
    while IFS= read -r config; do
      run_one "$config"
    done < <(find "$CONFIG_DIR/$method/$dataset" -name "*.json" 2>/dev/null | sort)
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
