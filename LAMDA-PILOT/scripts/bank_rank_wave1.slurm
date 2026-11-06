#!/bin/bash
#SBATCH --job-name=bank_rank_wave1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bank_rank_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bank_rank_wave1/slurm_%j.err
#
# Bank-effective-rank comparison campaign (2026-11-06 user request):
# O-LoRA and LoRA-Subtraction (growing-bank variant), each on its own
# vanilla/settled hyperparameters, across all 5 master-table datasets
# (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t, SUN397-10t,
# Food-101-20t), 3 seeds each (1993/1996/1999) = 30 runs.
#
# InfLoRA-Bank (originally a 3rd method here) was EXCLUDED 2026-11-06, per
# instruction, after the already-running 45-run version of this job showed
# InfLoRA-Bank's own 15 runs were far more expensive than expected -- its
# growing per-task bank makes its merged forward pass real dense-matmul
# work that gets slower every task (not an artifact), and just its first 2
# of 5 datasets (cifar224: 3 seeds at ~1h07m each; food101: 3 seeds at
# ~2h20m each, confirmed from that job's own SLURM log) had already
# consumed ~7h with 3 of 5 datasets and both other methods still to go --
# on pace to badly overrun a single 96h submission. See
# scripts/gen_bank_rank_wave1_configs.py's own docstring for the full
# writeup.
#
# Every config also sets log_bank_rank=true (utils/rank_diagnostics.py) --
# the summed-bank effective-rank diagnostic (singular values needed for 99%
# spectral energy of the cumulative per-task adapter update). Both methods'
# mechanics (training, memory accounting, this rank diagnostic) were
# smoke-tested locally before this campaign was built -- see the
# conversation record. NOT included: the sketch variants (InfLoRA-Sketch,
# LoRA-Subtraction-Sketch) -- this campaign is specifically about the
# GROWING-BANK methods' own summed-update rank, which isn't a meaningful
# question for a method whose bank is already rank-capped by construction
# (see models/lorasub_sketch.py's own deliberate no-op override of
# _log_bank_effective_rank).
#
# HYPERPARAMETERS (full detail in scripts/gen_bank_rank_wave1_configs.py's
# own docstring):
#   O-LoRA: identical to the existing master-table exps/
#     wave1_final_completion/olora_<dataset>_s1993.json configs.
#   LoRA-Subtraction: this session's own just-settled per-dataset LRs
#     (exps/lorasub_lr_tune/SWEEP_LOG.md) -- no prior "wave1_final"
#     convention exists for this method. use_proto_repulsion=true is
#     REQUIRED (the catastrophic-collapse fix found earlier this session),
#     not optional.
#
# Config generator: scripts/gen_bank_rank_wave1_configs.py (run at the top
# of this script, same as every other campaign here -- configs are
# regenerated, not hand-committed).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <model_name>/metrics_<tag>.json's own status=="done" -- these are all
# ordinary oracle-mode runs, final_metrics=true throughout, so this is the
# right checker, NOT scripts/_stream_run_done.py, which only applies to
# boundary_mode streaming drivers).
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every
# other campaign script in this project (this cluster has exactly one GPU
# available).
#
# 96h wall time kept as-is despite dropping a method -- OmniBenchmark-1k-
# 100t x2 methods x3 seeds = 6 runs are still the most expensive runs in
# this set and neither method's cost on it has been measured yet; the
# margin is intentional, not a sign this is expected to need a second
# submission (unlike the original 45-run version this replaces).
#
# A retry's plain `>` would silently overwrite a prior failed attempt's
# log -- run_one() below backs up any existing outfile first (same
# convention as every other campaign script here).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=bank_rank_wave1
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
python scripts/gen_bank_rank_wave1_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. olora/cifar224
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

# Run order: cheapest-feeling method first within each dataset isn't
# knowable ahead of time (both are full-length oracle runs of similar
# cost), so just go method-major, dataset-minor, `sort` keeping seed order
# deterministic within each. OmniBenchmark-1k left to run wherever it falls
# alphabetically within each method (no special-casing -- unlike some
# other campaigns here, there's no reason to defer it, since every run in
# this set is already full-length).
echo "==== ${CAMPAIGN} start: 30 runs, single GPU $(date) ===="

for method in olora lorasub; do
  for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
    while IFS= read -r config; do
      run_one "$config"
    done < <(find "$CONFIG_DIR/$method/$dataset" -name "*.json" 2>/dev/null | sort)
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
