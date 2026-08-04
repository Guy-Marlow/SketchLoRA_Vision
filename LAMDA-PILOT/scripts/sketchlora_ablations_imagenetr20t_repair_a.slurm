#!/bin/bash
#SBATCH --job-name=sketchlora_ablations_imagenetr20t_repair_a
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/repair_a_slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/repair_a_slurm_%j.err
#
# Repair/extension round for sketchlora_ablations_imagenetr20t (user request
# 2026-08-04/05, following analysis of job 22181's results -- see
# sketchlora_ablations_imagenetr20t.md), PART A of 2 (user request 2026-08-05:
# split the original single repair script in two, same resources each --
# see PART B, scripts/sketchlora_ablations_imagenetr20t_repair_b.slurm, for
# the other half). SAME campaign directory/conventions as
# scripts/sketchlora_ablations_imagenetr20t.slurm (ImageNet-R full 20-task
# split, oracle boundaries, final_metrics=true, ce_profile_every=0) -- this
# script does NOT re-run or touch that script's own 6 variants x 3 seeds
# (current/exactsvd/globaleps/fixedrank/countsketch stay exactly as they are;
# countsketch_s1996/s1999, still incomplete from job 22181, are OUT OF SCOPE
# here -- resubmit the original script separately for those).
#
# This part (A) runs 3 variants x 3 seeds = 9 cells:
#
#   1. exactsvd_ca (REPAIRED, FORCE-RERUN) -- job 22181's exactsvd_ca cells
#      produced byte-identical accuracy curves to plain exactsvd on every
#      seed: classifier_alignment was DEAD CODE under oracle-mode training
#      (_stream_end_chunk/_stream_init, where CA's actual effect lived, are
#      StreamMixin hooks only ever invoked by bounded_memory_mixin.py's/
#      stream_mixin.py's own drivers -- never by trainer.py's plain per-task
#      loop this whole campaign uses). Fixed 2026-08-05 in
#      models/sketchlora.py (_ca_lazy_init_stats/_ca_reset_reservoir/
#      _run_ca_alignment/_train_with_ca -- see that file's own "2026-08-05
#      fix" comments on _train). The CONFIG is unchanged (already correct);
#      only the code was broken. This is the ONE cell in this script that
#      bypasses the usual "status": done skip -- see FORCE_RERUN below -- so
#      it OVERWRITES run_logs/final/sketchlora/{metrics,ops_ledger}_
#      sketchlora_imagenetr_sketchlora_ablations_imagenetr20t_exactsvd_ca_
#      s<seed>_s<seed>.json in place. Its own .out log is written as
#      exactsvd_ca_s<seed>.out (NOT sketchlora_exactsvd_ca_s<seed>.out, which
#      is job 22181's original -- CORRUPTED -- capture; kept side-by-side on
#      purpose as a historical record, not overwritten).
#   2. seqlora -- SeqLoRA on the full 20-task split, for direct comparison
#      against the SketchLoRA family (this campaign never included it).
#      Hyperparameters copied verbatim from the project's one standing
#      SeqLoRA convention (scripts/gen_imagenetr_slurm_grid_configs.py's
#      METHOD_CFG).
#   3. noadapt -- the no-adaptation baseline (models/noadapt.py, added
#      alongside this script): frozen ViT-B/16 backbone (LoRA stays at its
#      zero-init no-op state, never trained), classifier head fit by a
#      single closed-form nearest-class-mean pass per task -- zero gradient
#      steps of any kind. Pure CIL evaluation of "how good is the frozen
#      backbone alone" -- much cheaper than the other cells (no backward
#      pass, no multi-epoch loop).
#
# Config generation: scripts/gen_sketchlora_ablations_imagenetr20t_repair_
# configs.py -- writes into the SAME exps/sketchlora_ablations_imagenetr20t/
# directory as the original generator, ADDING new files only (seqlora_s<seed>
# .json, noadapt_s<seed>.json, fixedrank_ca_s<seed>.json,
# fixedrank_exactsvd_ca_s<seed>.json -- the latter two are PART B's, this
# script generates all 12 anyway since the generator is a single idempotent
# script shared by both parts; harmless if PART B's own run regenerates the
# same files again, byte-identical output). Does not regenerate or touch
# exactsvd_ca's own config (unchanged -- the bug was in code, not config).
#
# OUTPUT FILES land in run_logs/sketchlora_ablations_imagenetr20t/ (the SAME
# directory job 22181's original 6-variant campaign used), named
# <variant>_s<seed>.out -- seqlora_s<seed>.out and noadapt_s<seed>.out are
# BRAND NEW filenames (no existing file of that name in that directory), so
# nothing is overwritten; exactsvd_ca_s<seed>.out is likewise new (distinct
# from job 22181's sketchlora_exactsvd_ca_s<seed>.out, see above). The two
# repaired/updated run_logs/final/sketchlora/{metrics,ops_ledger}_..._
# exactsvd_ca_..._s<seed>.json files ARE intentionally overwritten in place
# (that's the repair) -- run_logs/final/seqlora/ and run_logs/final/noadapt/
# are brand-new directories (MetricsLogger creates them via os.makedirs).
#
# NAMING (confirmed against trainer.py's oracle-path tag construction,
# "{model_name}_{dataset}_{prefix}_s{seed}" -- model_name varies per variant
# here, see MODEL_NAME below):
#   config  = exps/sketchlora_ablations_imagenetr20t/${CONFIG_FNAME}_s${seed}.json
#   prefix  = sketchlora_ablations_imagenetr20t_${variant}_s${seed}   (set inside the config)
#   tag     = ${model_name}_imagenetr_${prefix}_s${seed}
#   metrics_json = run_logs/final/${model_name}/metrics_${tag}.json
#   ledger_json  = run_logs/final/${model_name}/ops_ledger_${tag}.json
#
# RESOURCES / WALL TIME: 1 GPU, 10 CPU cores, 48GB RAM (identical to PART B
# and to the original ablations script), 3h30m (explicit user request
# 2026-08-05). Rough estimate from job 22181's own per-cell timings:
# exactsvd_ca-with-CA-now-actually-running ~35-40min x 3 (~105-120min),
# seqlora ~27min x 3 (~81min), noadapt much cheaper (no backward pass,
# ~5min x 3, ~15min) -- total ~200-215min, close to the 3h30m budget with
# some margin. RESUMABLE: same lockfile + skip-if-"status":done convention as
# every other script in this series, EXCEPT exactsvd_ca (see FORCE_RERUN
# below) -- once this script's own force-rerun produces a fresh "status":
# done for exactsvd_ca (tracked via the .repaired_exactsvd_ca_s<seed> marker
# file), a resubmission within the same repair round will skip it like
# anything else.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/sketchlora_ablations_imagenetr20t

LOCKFILE="run_logs/sketchlora_ablations_imagenetr20t/.sketchlora_ablations_imagenetr20t_repair_a.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another sketchlora_ablations_imagenetr20t_repair_a.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check + NGPU derivation (same convention as
# sketchlora_ablations_imagenetr20t.slurm).
NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>/tmp/sketchlora_ablations_imagenetr20t_repair_a_preflight_$$.err)
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat /tmp/sketchlora_ablations_imagenetr20t_repair_a_preflight_$$.err
  rm -f /tmp/sketchlora_ablations_imagenetr20t_repair_a_preflight_$$.err
  exit 1
fi
rm -f /tmp/sketchlora_ablations_imagenetr20t_repair_a_preflight_$$.err
echo "==== detected NGPU=${NGPU} (torch.cuda.device_count(), derived not hardcoded) $(date) ===="

# Same ImageNet-R data-path symlink fix as ce_smoke_imagenetr5t.slurm /
# imagenetr_slurm_grid.slurm (commit 14257dd) -- utils/data.py hardcodes
# ./data/imagenet-r/{train,test} (hyphenated), this cluster materializes it
# at ./data/imagenetr (no hyphen).
if [ ! -e ./data/imagenet-r ] && [ -d ./data/imagenetr ]; then
  ln -s imagenetr ./data/imagenet-r
  echo "==== symlinked ./data/imagenet-r -> ./data/imagenetr $(date) ===="
fi

echo "==== config generation start $(date) ===="
python scripts/gen_sketchlora_ablations_imagenetr20t_repair_configs.py
echo "==== config generation done $(date) ===="

SEEDS=(1993 1996 1999)

# variant -> config filename stem (exps/sketchlora_ablations_imagenetr20t/${CONFIG_FNAME}_s${seed}.json)
declare -A CONFIG_FNAME=(
  [exactsvd_ca]=sketchlora_exactsvd_ca   # from the ORIGINAL generator -- unchanged, reused as-is
  [seqlora]=seqlora
  [noadapt]=noadapt
)
# variant -> model_name (determines run_logs/final/<model_name>/ output dir + tag prefix)
declare -A MODEL_NAME=(
  [exactsvd_ca]=sketchlora
  [seqlora]=seqlora
  [noadapt]=noadapt
)
# variant -> 1 if a PRE-EXISTING "status": done marker (from before this
# script ever ran) should be ignored and the cell forced to rerun once.
declare -A FORCE_RERUN=(
  [exactsvd_ca]=1
)

VARIANTS=(exactsvd_ca seqlora noadapt)

echo "==== sketchlora_ablations_imagenetr20t_repair_a start: ${#VARIANTS[@]} variants x ${#SEEDS[@]} seeds, ${NGPU} GPU(s) $(date) ===="

run_one() {
  # $1=variant $2=seed
  local variant="$1" seed="$2"
  local model_name="${MODEL_NAME[$variant]}"
  local config="exps/sketchlora_ablations_imagenetr20t/${CONFIG_FNAME[$variant]}_s${seed}.json"
  local prefix="sketchlora_ablations_imagenetr20t_${variant}_s${seed}"
  local tag="${model_name}_imagenetr_${prefix}_s${seed}"
  local metrics_json="run_logs/final/${model_name}/metrics_${tag}.json"
  local out_tag="${variant}_s${seed}"   # .out filename in run_logs/sketchlora_ablations_imagenetr20t/ -- new for all 3 variants here, nothing pre-existing shares these names

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${variant}_s${seed} $(date) ===="
    return
  fi

  local force_marker="run_logs/sketchlora_ablations_imagenetr20t/.repaired_${variant}_s${seed}"
  if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    if [ "${FORCE_RERUN[$variant]:-0}" = "1" ] && [ ! -f "$force_marker" ]; then
      echo "==== FORCE-RERUN (overwriting pre-repair output) ${variant}_s${seed} $(date) ===="
    else
      echo "==== SKIP (done) ${variant}_s${seed} $(date) ===="
      return
    fi
  fi

  echo "==== START ${variant}_s${seed} (model=${model_name}) $(date) ===="
  python main.py --config "$config" \
    > "run_logs/sketchlora_ablations_imagenetr20t/${out_tag}.out" 2>&1 || true
  touch "$force_marker"
  echo "==== DONE  ${variant}_s${seed} $(date) ===="
}

for VARIANT in "${VARIANTS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "$VARIANT" "$SEED"
  done
done

echo "==== PART A COMPLETE (or wall reached) $(date) ===="
echo "==== outputs: run_logs/final/{sketchlora,seqlora,noadapt}/{metrics,ops_ledger}_*.json, .out logs under run_logs/sketchlora_ablations_imagenetr20t/ ===="
