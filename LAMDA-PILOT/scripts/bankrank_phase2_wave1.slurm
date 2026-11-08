#!/bin/bash
#SBATCH --job-name=bankrank_phase2_wave1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bankrank_phase2_wave1/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/bankrank_phase2_wave1/slurm_%j.err
#
# Bank-rank phase 2 campaign (2026-11-09 user request): three sequential
# sub-campaigns, in this order -- (1) vanilla SD-LoRA with log_bank_rank=true
# (models/sdlora.py's own summed-bank effective-rank diagnostic, added this
# session), (2) vanilla CL-LoRA with log_bank_rank=true (models/cllora.py's
# own diagnostic, also added this session -- a HYPOTHETICAL "if the per-task
# specific_pos adapters were folded into one dense update" capacity proxy,
# since CL-LoRA's real forward never sums them, see that file's docstring),
# (3) CL-LoRA-Sketch (the "v3" orth_a_side-only design from the earlier
# smoke-scale ablation, exps/cllora_sketch_wave1/v3_orth_only_s1993.json,
# promoted to a full-length 5-dataset x 3-seed campaign) with
# sketchlora_align_weight RAISED to 0.75 (was 0.5 at smoke scale) per
# explicit user instruction -- no log_bank_rank here, this phase is about
# accuracy/memory at full length, not the rank diagnostic.
#
# 5 datasets (CIFAR-100-10t, ImageNet-R-20t, OmniBenchmark-1k-100t,
# SUN397-10t, Food-101-20t) x 3 seeds (1993/1996/1999) x 3 sub-campaigns =
# 45 runs.
#
# Both diagnostics (SD-LoRA's and CL-LoRA's) were smoke-tested locally
# before this script was written -- see the conversation record (3-task
# smoke runs, both produced sane, monotonically-growing effective-rank
# values with no errors).
#
# HYPERPARAMETERS (full detail in scripts/gen_bankrank_phase2_wave1_configs.py's
# own docstring):
#   sdlora: identical to exps/slao_sdlora_wave1_datasets/sdlora/<dataset>_
#     s1993.json (this project's own settled vanilla SD-LoRA convention),
#     plus log_bank_rank=true, bank_rank_energy_target=0.99.
#   cllora: identical to exps/wave1_final_completion/cllora_<dataset>_
#     s1993.json (this project's own settled vanilla CL-LoRA convention),
#     plus log_bank_rank=true, bank_rank_energy_target=0.99.
#   cllora_sketch: same architecture constants as cllora (ffn_num=10,
#     msa/general_pos/specfic_pos, 20 epochs/task), plus svd_rank=10,
#     merge_op="qrsvd", orth_a_side=true, use_block_weight_scaling=false,
#     sketchlora_align_weight=0.75. Full task count (no stop_after_tasks,
#     unlike the 5-task ablation smoke runs this config is promoted from).
#
# Config generator: scripts/gen_bankrank_phase2_wave1_configs.py (run at the
# top of this script, same as every other campaign here -- configs are
# regenerated, not hand-committed).
#
# RESUMABLE via scripts/_bankcap_run_done.py (checks run_logs/final/
# <model_name>/metrics_<tag>.json's own status=="done") -- all three
# sub-campaigns are ordinary oracle-mode runs, final_metrics=true throughout.
#
# SINGLE GPU BY CONSTRUCTION -- same plain sequential pattern as every other
# campaign script in this project (this cluster has exactly one GPU
# available).
#
# 96h wall time: 45 full-length runs including OmniBenchmark-1k-100t x3
# sub-campaigns x3 seeds = 9 of the most expensive runs in this set (CL-LoRA
# alone measured at ~12.4h/seed on that split in this project's own prior
# master-table run) -- this will almost certainly need multiple
# resubmissions, same as every other long campaign here; the RESUMABLE
# design above is what makes that safe, not a sign this is mis-scoped.
#
# A retry's plain `>` would silently overwrite a prior failed attempt's
# log -- run_one() below backs up any existing outfile first (same
# convention as every other campaign script here).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=bankrank_phase2_wave1
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
python scripts/gen_bankrank_phase2_wave1_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json.
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local subdir
  subdir="$(dirname "$config" | sed "s#^${CONFIG_DIR}/##")"   # e.g. sdlora/cifar224
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

# Run order: sdlora, then cllora, then cllora_sketch (exactly the order the
# user asked for) -- dataset-minor within each, `sort` keeping seed order
# deterministic. No special-casing OmniBenchmark-1k -- every run in this set
# is already full-length.
echo "==== ${CAMPAIGN} start: 45 runs, single GPU $(date) ===="

for method in sdlora cllora cllora_sketch; do
  for dataset in cifar224 food101 imagenetr omnibenchmark1k sun397; do
    while IFS= read -r config; do
      run_one "$config"
    done < <(find "$CONFIG_DIR/$method/$dataset" -name "*.json" 2>/dev/null | sort)
  done
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE (or wall reached) $(date) ===="
