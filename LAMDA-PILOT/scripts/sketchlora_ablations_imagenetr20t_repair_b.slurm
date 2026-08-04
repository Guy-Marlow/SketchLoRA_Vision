#!/bin/bash
#SBATCH --job-name=sketchlora_ablations_imagenetr20t_repair_b
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/repair_b_slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/repair_b_slurm_%j.err
#
# Repair/extension round for sketchlora_ablations_imagenetr20t (user request
# 2026-08-04/05, following analysis of job 22181's results -- see
# sketchlora_ablations_imagenetr20t.md), PART B of 2 (user request 2026-08-05:
# split the original single repair script in two, same resources each --
# see PART A, scripts/sketchlora_ablations_imagenetr20t_repair_a.slurm, for
# the other half: exactsvd_ca force-rerun + seqlora + noadapt). SAME campaign
# directory/conventions as scripts/sketchlora_ablations_imagenetr20t.slurm
# (ImageNet-R full 20-task split, oracle boundaries, final_metrics=true,
# ce_profile_every=0) -- this script does NOT re-run or touch that script's
# own 6 variants x 3 seeds (current/exactsvd/globaleps/fixedrank/countsketch
# stay exactly as they are; countsketch_s1996/s1999, still incomplete from
# job 22181, are OUT OF SCOPE here -- resubmit the original script separately
# for those). Independent of PART A -- no shared state beyond the config
# generator and the output directory, safe to submit both at once.
#
# This part (B) runs 2 variants x 3 seeds = 6 cells, both NEW (never run
# before, ordinary skip-if-done, no force-rerun needed):
#
#   1. fixedrank_ca -- fixedrank (rank pinned at svd_rank=10, no adaptive
#      threshold, merge_op=randsvd, admission=global_eps -- identical to the
#      existing "fixedrank" variant) PLUS classifier_alignment=True (same
#      ca_steps=300/ca_batch=128/ca_lr=0.001 as exactsvd_ca). Isolates how
#      much of fixedrank's ~3.6pt deficit vs. "current" (see
#      sketchlora_ablations_imagenetr20t.md) is closable by correcting
#      classifier HEAD drift alone, vs. requiring the adaptive rank itself.
#      Requires the SAME 2026-08-05 classifier_alignment/oracle-mode fix
#      PART A's exactsvd_ca repair depends on (models/sketchlora.py) -- this
#      variant never ran under the old broken code at all, so there is no
#      "corrupted" prior output to worry about overwriting.
#   2. fixedrank_exactsvd_ca -- fixedrank_ca but merge_op=exactsvd instead of
#      randsvd, to see how much FURTHER ground exact-SVD reconstruction
#      recovers once CA has already been given credit.
#
# Config generation: scripts/gen_sketchlora_ablations_imagenetr20t_repair_
# configs.py -- SAME script PART A also runs; idempotent (regenerates all 12
# files byte-identically), so running it again here is harmless even if PART
# A already did.
#
# OUTPUT FILES land in run_logs/sketchlora_ablations_imagenetr20t/ (the SAME
# directory job 22181's original 6-variant campaign, and PART A, both use),
# named <variant>_s<seed>.out -- fixedrank_ca_s<seed>.out and
# fixedrank_exactsvd_ca_s<seed>.out are BRAND NEW filenames (no existing file
# of that name anywhere in that directory, including PART A's own outputs),
# so nothing from PART A, job 22181, or this script's own reruns gets
# overwritten. run_logs/final/sketchlora/ already exists (shared with every
# other sketchlora variant, including PART A's exactsvd_ca) -- this script
# only ever writes files tagged fixedrank_ca/fixedrank_exactsvd_ca there,
# distinct tags from anything else that's ever been written to that
# directory.
#
# NAMING (identical convention to PART A -- see that script's own header for
# the full derivation against trainer.py):
#   config  = exps/sketchlora_ablations_imagenetr20t/${variant}_s${seed}.json
#   prefix  = sketchlora_ablations_imagenetr20t_${variant}_s${seed}   (set inside the config)
#   tag     = sketchlora_imagenetr_${prefix}_s${seed}    (model_name=sketchlora for both cells here)
#   metrics_json = run_logs/final/sketchlora/metrics_${tag}.json
#   ledger_json  = run_logs/final/sketchlora/ops_ledger_${tag}.json
#
# RESOURCES / WALL TIME: 1 GPU, 10 CPU cores, 48GB RAM (identical to PART A
# and to the original ablations script), 3h30m (explicit user request
# 2026-08-05). Rough estimate from job 22181's own per-cell timings (a
# CA-bearing sketchlora cell runs somewhat longer than its non-CA
# counterpart, ~35-40min each): 6 cells x ~35-40min =~ 210-240min, close to
# or slightly over the 3h30m budget -- RESUMABLE design (skip-if-
# "status":done + lockfile) means a resubmission picks up wherever this run
# left off if the wall clock is hit.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/sketchlora_ablations_imagenetr20t

LOCKFILE="run_logs/sketchlora_ablations_imagenetr20t/.sketchlora_ablations_imagenetr20t_repair_b.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another sketchlora_ablations_imagenetr20t_repair_b.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check + NGPU derivation (same convention as
# sketchlora_ablations_imagenetr20t.slurm / PART A).
NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>/tmp/sketchlora_ablations_imagenetr20t_repair_b_preflight_$$.err)
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat /tmp/sketchlora_ablations_imagenetr20t_repair_b_preflight_$$.err
  rm -f /tmp/sketchlora_ablations_imagenetr20t_repair_b_preflight_$$.err
  exit 1
fi
rm -f /tmp/sketchlora_ablations_imagenetr20t_repair_b_preflight_$$.err
echo "==== detected NGPU=${NGPU} (torch.cuda.device_count(), derived not hardcoded) $(date) ===="

# Same ImageNet-R data-path symlink fix as ce_smoke_imagenetr5t.slurm /
# imagenetr_slurm_grid.slurm / PART A (commit 14257dd) -- utils/data.py
# hardcodes ./data/imagenet-r/{train,test} (hyphenated), this cluster
# materializes it at ./data/imagenetr (no hyphen). Idempotent -- a no-op if
# PART A (or any other script) already created the symlink.
if [ ! -e ./data/imagenet-r ] && [ -d ./data/imagenetr ]; then
  ln -s imagenetr ./data/imagenet-r
  echo "==== symlinked ./data/imagenet-r -> ./data/imagenetr $(date) ===="
fi

echo "==== config generation start $(date) ===="
python scripts/gen_sketchlora_ablations_imagenetr20t_repair_configs.py
echo "==== config generation done $(date) ===="

SEEDS=(1993 1996 1999)
VARIANTS=(fixedrank_ca fixedrank_exactsvd_ca)

echo "==== sketchlora_ablations_imagenetr20t_repair_b start: ${#VARIANTS[@]} variants x ${#SEEDS[@]} seeds, ${NGPU} GPU(s) $(date) ===="

run_one() {
  # $1=variant $2=seed -- both cells here are model_name=sketchlora
  local variant="$1" seed="$2"
  local config="exps/sketchlora_ablations_imagenetr20t/${variant}_s${seed}.json"
  local prefix="sketchlora_ablations_imagenetr20t_${variant}_s${seed}"
  local tag="sketchlora_imagenetr_${prefix}_s${seed}"
  local metrics_json="run_logs/final/sketchlora/metrics_${tag}.json"
  local out_tag="${variant}_s${seed}"   # .out filename in run_logs/sketchlora_ablations_imagenetr20t/ -- new, no collision anywhere in that directory

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${variant}_s${seed} $(date) ===="
    return
  fi
  if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done) ${variant}_s${seed} $(date) ===="
    return
  fi

  echo "==== START ${variant}_s${seed} $(date) ===="
  python main.py --config "$config" \
    > "run_logs/sketchlora_ablations_imagenetr20t/${out_tag}.out" 2>&1 || true
  echo "==== DONE  ${variant}_s${seed} $(date) ===="
}

for VARIANT in "${VARIANTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "$VARIANT" "$SEED"
  done
done

echo "==== PART B COMPLETE (or wall reached) $(date) ===="
echo "==== outputs: run_logs/final/sketchlora/{metrics,ops_ledger}_*.json, .out logs under run_logs/sketchlora_ablations_imagenetr20t/ ===="
