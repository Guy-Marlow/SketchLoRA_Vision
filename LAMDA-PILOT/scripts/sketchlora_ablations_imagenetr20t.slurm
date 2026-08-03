#!/bin/bash
#SBATCH --job-name=sketchlora_ablations_imagenetr20t
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/sketchlora_ablations_imagenetr20t/slurm_%j.err
#
# SketchLoRA compression-scheme ablation series (user request 2026-07-31,
# extended 2026-08-04 with two more variants): ORDINARY task-incremental
# training (no boundary_mode -- real task boundaries, no memory-budget
# streaming), ImageNet-R full 20-task split, 6 variants x 3 seeds = 18 runs.
# See scripts/gen_sketchlora_ablations_imagenetr20t_configs.py's own
# docstring for the exact per-variant config and the reasoning behind each
# (one axis changed per variant off the "current"/frozen-v1 baseline):
#   current     -- the exact config already running throughout round2_slurm_
#                  grid/imagenetr_slurm_grid (adaptive threshold,
#                  bounded_eviction, rank_cap=128, randsvd).
#   exactsvd    -- current, but merge_op=exactsvd instead of randsvd.
#   globaleps   -- current, but sketchlora_admission=global_eps instead of
#                  bounded_eviction -- disables the eviction-count cap that
#                  stops a merge from removing more than one adapter's worth
#                  (lora_rank) of singular directions at once, so rank can now
#                  decrease between merges ("rank collapse" / retroactive
#                  mass-eviction, per models/sketchlora.py's own comment on
#                  the bounded_eviction rule it exists to fix). rank_cap=128
#                  stays on (global_eps still respects it as a plain clamp) --
#                  this isolates the eviction-cap axis alone. NOT the same
#                  axis as fd_shrinkage, which shrinks the magnitude of
#                  already-kept singular values and is untouched by any
#                  variant in this series.
#   fixedrank   -- svd_energy_target unset (fixed-rank path: r_hat_t is always
#                  svd_rank=lora_rank=10, never grows/shrinks/adapts);
#                  sketchlora_admission forced to "global_eps" (the only legal
#                  value once energy_target is None -- bounded_eviction/floor
#                  both assert energy_target is not None at construction,
#                  confirmed against models/sketchlora.py before this script
#                  was written). rank_cap=128 kept per instruction, confirmed
#                  inert (fixed rank never approaches it).
#   exactsvd_ca -- exactsvd (current + merge_op=exactsvd), PLUS
#                  classifier_alignment=True (ca_steps=300/ca_batch=128/
#                  ca_lr=0.001, copied from the only other completed CA run
#                  in this project, exps/sketchlora_boltons/100mb_ca.json).
#   countsketch -- fixedrank's sibling, NOT current's (user correction,
#                  2026-08-04): svd_energy_target unset, same as fixedrank,
#                  REQUIRED here since CountSketch has no singular-value-like
#                  signal to drive an adaptive rank threshold the way randsvd/
#                  exactsvd's spectrum does -- choosing r_hat_t via the
#                  SVD-based path and then discarding that SVD's basis for a
#                  random hash merge would smuggle exact-SVD rank information
#                  into a variant meant to isolate hashing WITHOUT it.
#                  cs_rank omitted -> defaults to svd_rank=10, matching
#                  fixedrank's own fixed rank exactly, so this isolates the
#                  merge-ALGORITHM axis alone (hash-based CountSketch vs SVD
#                  truncation) at the same rank fixedrank uses. Audited
#                  utils/countsketch.py directly before adding this: the
#                  hash-merge implementation itself looks sound -- handles
#                  the all-zero-norm edge case, uses a deterministic
#                  per-(task,module) seed, no bugs found.
#
# 3 seeds (1993/1996/1999) assumed from this project's standing convention --
# not explicitly specified in the request; flag if a different set was meant.
#
# final_metrics=true in every config (the oracle path's actual MetricsLogger
# gate, unlike bounded_memory where it's unconditional) -- gives persistent-
# memory/FLOP/train-eval-seconds data comparable to the rest of this project.
# STALE NOTE CORRECTED 2026-08-04: trainer.py's oracle path DOES now write a
# CE (ops_ledger) record every task too (added alongside utils/ce_profiler.py,
# commit 886455b) -- final_metrics also gates that. What it does NOT give,
# by design here: the torch.profiler REGION breakdown (measured_step_regions/
# measured_boundary_regions) -- ce_profile_every=0 (see the config generator's
# own docstring) disables that specifically, since ce_smoke_imagenetr5t
# already showed it's expensive enough to make a 20-task run impractical.
# The formula-based ops_total/ops_fb and the cheap single-batch R2 baseline
# probe (ops_fb_baseline, merged_forward_excess_per_step) are unaffected and
# still populate every record.
#
# Wall time set to 12h (explicit user request 2026-08-04, down from a rough
# 144h full-completion estimate for all 18 runs sequentially on one GPU) --
# at this budget the job will almost certainly NOT finish all 18 cells in one
# submission; it relies on the RESUMABLE/LOCKFILE design below (skip whatever
# already shows "status": "done") to make partial progress across repeated
# resubmissions rather than needing one long-lived allocation. Whatever
# variant/seed is mid-run when the wall clock hits will be left incomplete
# (no partial-epoch checkpoint inside a single task) and simply re-run from
# scratch on the next submission -- exactsvd_ca (extra ca_steps=300 per task
# boundary) and the slower merge_ops are likeliest to be the ones still
# incomplete after any given 12h window.
#
# RESOURCES (1 GPU, 10 CPU cores, 48GB RAM -- explicit user request
# 2026-08-04, down from the prior 2 GPU/18 cpu/128GB): NGPU below is DERIVED
# from what SLURM actually granted (torch.cuda.device_count(), read once
# during the pre-flight check), not hardcoded -- so --gres in the header
# above is now the ONLY place GPU count is specified; the worker-pool code
# further down automatically matches whatever that says, including if it's
# ever changed back up. This was a real bug in the prior version of this
# script: NGPU=2 was hardcoded independently of --gres, so requesting fewer
# GPUs than 2 without also editing NGPU would have silently sent real jobs to
# a nonexistent CUDA device (see prior conversation turn for the full
# failure mode) -- structurally can't happen anymore.
#
# RESUMABLE / LOCKFILE: same convention as every other script in this
# series -- skips a cell whose accuracy json shows "partial": false; refuses
# to start a second instance of itself. Oracle-path completion signal is
# DIFFERENT from bounded_memory's (see run_one() below) -- there is no
# "boundedmem_*.json"/"partial" field for oracle runs; completion is read
# from the metrics json's "status": "done" alone (final_metrics=true is what
# makes that file exist at all for this path).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/sketchlora_ablations_imagenetr20t

LOCKFILE="run_logs/sketchlora_ablations_imagenetr20t/.sketchlora_ablations_imagenetr20t.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another sketchlora_ablations_imagenetr20t.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Pre-flight sanity check (same convention as round2_slurm_grid.slurm),
# EXTENDED to also derive NGPU here (2026-08-04, user request) -- one
# python/torch startup does both jobs. torch.cuda.device_count() reads
# however many GPUs are actually visible inside this job's cgroup, which is
# whatever --gres above actually granted, NOT a value duplicated/hardcoded
# in this file -- so the two can never drift apart again (see the header
# comment on the bug this replaces).
NGPU=$(python -c "import torch, timm; assert torch.cuda.is_available(); print(torch.cuda.device_count())" \
  2>/tmp/sketchlora_ablations_imagenetr20t_preflight_$$.err)
# Regex validation, not a numeric [ -lt ] comparison: if NGPU somehow came
# back non-numeric, "[ "$NGPU" -lt 1 ]" itself errors ("integer expression
# expected") with a NONZERO exit status -- which reads as FALSE to this if,
# i.e. the FATAL branch would be silently skipped and the script would carry
# on with a garbage NGPU. Matching a strict positive-integer pattern instead
# has no such failure mode.
if [ -z "$NGPU" ] || ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: pre-flight check failed (torch/timm import, CUDA unavailable, or non-numeric device count). Details:"
  cat /tmp/sketchlora_ablations_imagenetr20t_preflight_$$.err
  rm -f /tmp/sketchlora_ablations_imagenetr20t_preflight_$$.err
  exit 1
fi
rm -f /tmp/sketchlora_ablations_imagenetr20t_preflight_$$.err
echo "==== detected NGPU=${NGPU} (torch.cuda.device_count(), derived not hardcoded) $(date) ===="

echo "==== config generation start $(date) ===="
python scripts/gen_sketchlora_ablations_imagenetr20t_configs.py
echo "==== config generation done $(date) ===="

VARIANTS=(current exactsvd globaleps fixedrank exactsvd_ca countsketch)
SEEDS=(1993 1996 1999)

echo "==== sketchlora_ablations_imagenetr20t start: ${#VARIANTS[@]} variants x ${#SEEDS[@]} seeds, ${NGPU} GPUs $(date) ===="

run_one() {
  # $1=gpu $2=variant $3=seed
  local gpu="$1" variant="$2" seed="$3"
  local tag="sketchlora_${variant}_s${seed}"
  local config="exps/sketchlora_ablations_imagenetr20t/${tag}.json"
  local prefix="sketchlora_ablations_imagenetr20t_${variant}_s${seed}"
  # oracle path (trainer.py:140-143, final_metrics gate): tag is
  # "{model_name}_{dataset}_{prefix}_s{seed}" -- NOT bounded_memory's
  # "{model_name}_{prefix}_s{seed}" (no dataset component) -- confirmed
  # against trainer.py source directly, not assumed from the other script's
  # convention. Only the metrics json exists for this path; there is no
  # boundedmem_*.json / "partial" flag (that's bounded_memory_mixin.py-only).
  local metrics_tag="sketchlora_imagenetr_${prefix}_s${seed}"
  local metrics_json="run_logs/final/sketchlora/metrics_${metrics_tag}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START [gpu ${gpu}] ${tag} $(date) ===="
  CUDA_VISIBLE_DEVICES="${gpu}" python main.py --config "$config" \
    > "run_logs/sketchlora_ablations_imagenetr20t/${tag}.out" 2>&1 || true
  echo "==== DONE  [gpu ${gpu}] ${tag} $(date) ===="
}

JOBS_FIFO="run_logs/sketchlora_ablations_imagenetr20t/.jobs.fifo"
rm -f "$JOBS_FIFO"
mkfifo "$JOBS_FIFO"

(
  for VARIANT in "${VARIANTS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      echo "${VARIANT}|${SEED}"
    done
  done
) > "$JOBS_FIFO" &
FIFO_WRITER_PID=$!

worker() {
  local gpu="$1"
  local variant seed
  while IFS='|' read -r variant seed; do
    run_one "$gpu" "$variant" "$seed"
  done < "$JOBS_FIFO"
}

for (( g=0; g<NGPU; g++ )); do
  worker "$g" &
done
wait
wait "$FIFO_WRITER_PID" 2>/dev/null || true
rm -f "$JOBS_FIFO"

echo "==== ALL SKETCHLORA ABLATION RUNS COMPLETE (or wall reached) $(date) ===="
