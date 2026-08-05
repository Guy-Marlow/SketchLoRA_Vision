#!/bin/bash
#SBATCH --job-name=wave1_final
#SBATCH --time=192:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/wave1_final/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/wave1_final/slurm_%j.err
#
# Wave-1 final grid (user request 2026-08-05): SeqLoRA, SketchLoRA, O-LoRA,
# InfLoRA, TreeLoRA x {CIFAR-100-10t, ImageNet-R-20t, Food101-20t, SUN397-10t,
# OmniBenchmark-1k-100t} x 3 seeds (1993/1996/1999) = 75 runs, oracle CIL
# (scenario="cil", real task boundaries -- not bounded_memory streaming).
# Configs: exps/wave1_final/{method}_{dataset}_s{seed}.json, all 75 generated
# by scripts/gen_wave1_final_configs.py (idempotent -- called fresh below,
# safe to re-run). SketchLoRA's 5 configs were ADDED to that generator this
# session (previously only SeqLoRA/O-LoRA/InfLoRA/TreeLoRA existed here); its
# lr is 0.001 uniformly across all 5 datasets, per explicit user instruction,
# taken from its own completed 100-task OmniBenchmark-1k run and matching its
# already-live ImageNet-R config -- see that generator's own docstring for
# the full provenance, not re-derived here.
#
# SINGLE GPU BY CONSTRUCTION: this script requests exactly one GPU
# (--gres=gpu:1) and runs its 75-job queue as a single, strictly sequential
# loop -- no worker pool, no FIFO job-queue fan-out, nothing that assumes or
# benefits from a second device (unlike round2_slurm_grid.slurm/
# imagenetr_slurm_grid.slurm's NGPU-worker-pool idiom, built for their
# original 2-GPU allocations). CUDA_VISIBLE_DEVICES is pinned to 0 for every
# run; the preflight check below only confirms CUDA/torch/timm are usable, it
# does not size a worker pool off whatever device_count() returns.
#
# RUN ORDER (revised 2026-08-05, user request): TWO PHASES, not one flat
# method-major sweep -- OmniBenchmark-1k (100 tasks, ~7x ImageNet-R's task
# count, the dominant cost of this whole grid) is walled off into its own
# phase so it can never block or interleave with the four smaller benchmarks'
# results.
#   PHASE 1 (small datasets): SeqLoRA -> SketchLoRA -> O-LoRA -> InfLoRA ->
#     TreeLoRA, each completing all 3 seeds on cifar224 -> imagenetr ->
#     food101 -> sun397 before the next method starts. 60 runs.
#   PHASE 2 (OmniBenchmark-1k only): same method order, all 3 seeds each.
#     15 runs. Nothing in Phase 2 starts until every Phase 1 cell has been
#     attempted (not just started) -- see the two separate loops below; this
#     is a hard phase boundary, not a priority hint.
# GUARANTEE this buys: every method's results on the 4 smaller, cheaper
# benchmarks are secured before ANY GPU time goes to OmniBenchmark-1k, so a
# wall-clock cutoff (still plausible even at 8 days -- OmniBenchmark-1k alone
# is a large fraction of this grid's total cost) degrades gracefully to "all
# 5 methods have complete small-benchmark results, partial OmniBenchmark-1k
# coverage" rather than risking some methods having no OmniBenchmark-1k while
# also being mid-way through a small dataset. The resumable skip-check below
# (same convention as every other script in this project: skip a cell whose
# metrics json already reads "status": "done") means repeated resubmissions
# of this SAME script re-enter both phases in the same fixed order and skip
# whatever's already done -- this is a multi-submission design, not a promise
# of single-shot completion.
#
# WHAT'S LOGGED (user request, verified present before writing this script,
# not assumed):
#   - CIL top1/top5 accuracy curves: standard trainer.py accuracy logging,
#     unconditional on any flag here (print_forget=true additionally gets the
#     per-task forgetting breakdown printed). FIXED 2026-08-05: top5 used to
#     only reach the .out log (trainer.py's "CNN top5 curve" print) and NOT
#     the structured metrics_*.json (utils/metrics_logger.py::record_task
#     wrote cil_top1/til_top1 only) -- added cil_top5/til_top5 to that same
#     per-task record, so top5 no longer needs log-scraping to recover.
#   - SketchLoRA's sketch/LoRA parameters train with weight_decay=0.0
#     (`sketchlora_lora_wd=0.0` in its configs; models/sketchlora.py::
#     _optimizer_param_groups splits LoRA params into their own zero-decay
#     group once that field is set). The classifier head correctly keeps the
#     generic weight_decay=0.0005, same as every other method's head -- a
#     2026-08-05 attempt to zero the head's decay too was REVERTED same day
#     per direct user correction ("if it was already active on SketchLoRA's
#     classifier head, it should stay that way; we just don't want the
#     sketch decaying"). All 5 methods' heads use weight_decay=0.0005.
#   - Persistent memory: final_metrics=true (set in every config) gates
#     utils/metrics_logger.py's MetricsLogger, which calls each learner's own
#     persistent_state() every task. VERIFIED the accounting fixes are live in
#     the current tree, not just planned: O-LoRA (models/olora.py:229, counts
#     frozen_delta + un-folded old-A only, not every dead old-B) / InfLoRA
#     (models/inflora.py:64, frozen_delta + DualGPM bases + current slot, not
#     named_parameters()'s wrong default) / TreeLoRA (models/treelora.py:180,
#     frozen_delta + current slot + tree_grad_store, NOT every historical
#     adapter slot -- this one dropped a real run's reported figure from
#     1026MB to ~399MB when it was added) all override the naive generic
#     accounting with method-specific "what's actually still allocated" logic.
#     SeqLoRA and SketchLoRA use the shared generic persistent_state()
#     (models/lora.py:187, sums whatever adapter Linears currently exist) --
#     correct for both: SeqLoRA's footprint genuinely is "one adapter bank per
#     task, never freed" with no folding to account for, and SketchLoRA's
#     sketch slot is verified to be PHYSICALLY resized on every compress
#     (models/sketchlora.py::_resize_sketch_slot, not just logically relabeled
#     or padded to a worst-case max), so summing current parameter tensors
#     already gives the true O(r-hat*d) footprint at each point in time.
#   - Inference FLOPs: final_metrics=true also gates utils/flops.py's
#     measure_inference_flops -- works identically for all 5 methods (profiles
#     one real learner._deployed_forward call, no per-architecture formula).
#   - SketchLoRA reconstruction-error statistics: sketch_diag=true (set in its
#     5 configs) logs retained_energy/sigma_next/r_hat per compression, ALL
#     derived from the SAME rand_svd_probe() decomposition already computed to
#     pick r_hat_t (utils/randsvd.py) -- no second/extra SVD, confirmed by
#     reading models/sketchlora.py:1243-1257 directly. Written by the model to
#     run_logs/sketchlora_diag_<dataset>_adapt<eps>_ball_ic<ic>i<inc>_seed<seed>.json,
#     then MOVED by run_one() (below) to
#     run_logs/wave1_final/sketchlora_diag/ right after each run finishes --
#     see the CONSOLIDATION note on run_one() for why the move step exists.
#     FIXED THIS SESSION: that path previously keyed only on (energy_target,
#     n_lora_blocks, split, seed) -- NOT dataset -- so cifar224/imagenetr/
#     omnibenchmark1k, which all share the ic10i10 split shape in this exact
#     grid, would have silently overwritten each other's diagnostics (confirmed
#     colliding filenames already on disk from earlier campaigns). Dataset name
#     is now part of the path (models/sketchlora.py, self._diag_path); old
#     filenames are untouched, this only affects runs launched from now on --
#     i.e. it was necessary to fix BEFORE this script could safely produce
#     trustworthy reconstruction-error data across more than one dataset at a
#     shared split shape.
#
# RESUMABLE / LOCKFILE: same convention as every other script in this
# project -- skips a cell whose metrics json already shows "status": "done";
# refuses to start a second instance of itself.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/wave1_final

LOCKFILE="run_logs/wave1_final/.wave1_final.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another wave1_final.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Pre-flight sanity check ONLY -- unlike round2_slurm_grid.slurm/
# imagenetr_slurm_grid.slurm, this does NOT derive a worker-pool size from
# torch.cuda.device_count(); this script is single-GPU by construction (see
# header), so there is nothing to derive. A GPU count above 1 is not an
# error (the allocation may simply have more visible for other reasons) but
# is logged since it's unexpected for this job's --gres request.
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
  echo "NOTE: torch.cuda.device_count()=${NGPU} (expected 1 for this job's --gres=gpu:1) -- proceeding on CUDA_VISIBLE_DEVICES=0 regardless, this script never uses a second device."
fi
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

echo "==== config generation start $(date) ===="
python scripts/gen_wave1_final_configs.py
echo "==== config generation done $(date) ===="

# method-major order, exactly as requested -- do not alphabetize or reorder.
METHODS=(seqlora sketchlora olora inflora treelora)
SMALL_DATASETS=(cifar224 imagenetr food101 sun397)
BIG_DATASETS=(omnibenchmark1k)
SEEDS=(1993 1996 1999)

echo "==== wave1_final start: ${#METHODS[@]} methods x (${#SMALL_DATASETS[@]} small + ${#BIG_DATASETS[@]} big) datasets x ${#SEEDS[@]} seeds = $(( ${#METHODS[@]} * (${#SMALL_DATASETS[@]} + ${#BIG_DATASETS[@]}) * ${#SEEDS[@]} )) runs, single GPU, two phases $(date) ===="

run_one() {
  # $1=method $2=dataset $3=seed
  local method="$1" dataset="$2" seed="$3"
  local tag="${method}_${dataset}_s${seed}"
  local config="exps/wave1_final/${tag}.json"
  local prefix="wave1_final_${tag}"
  # oracle path (trainer.py: _tag = "{model_name}_{dataset}_{prefix}_s{seed}"),
  # same formula every other final_metrics-gated script in this project uses;
  # prefix itself already ends in _s<seed> (see gen_wave1_final_configs.py),
  # so the seed suffix is doubled in the metrics filename -- expected, matches
  # the existing convention elsewhere (e.g. sketchlora_ablations_imagenetr20t).
  local metrics_tag="${method}_${dataset}_${prefix}_s${seed}"

  # CONSOLIDATION (2026-08-05 user request, for easy scp back to this
  # machine): trainer.py/utils/metrics_logger.py/utils/ops_ledger.py hardcode
  # their own output directory (run_logs/final/<model_name>/, SHARED across
  # every campaign in this project, not configurable per-script) and
  # models/sketchlora.py's sketch_diag writes straight to top-level
  # run_logs/ -- none of that lands under run_logs/wave1_final/ on its own.
  # Every run here writes to those original locations first (can't change
  # that without touching shared infra every other campaign also depends on),
  # then gets MOVED into a dedicated run_logs/wave1_final/ subtree right
  # after it finishes. The skip-check below reads the MOVED (destination)
  # path, not the original -- so resubmissions still work correctly once a
  # cell's files have been relocated.
  local final_dir="run_logs/wave1_final/final/${method}"
  local diag_dir="run_logs/wave1_final/sketchlora_diag"
  mkdir -p "$final_dir"
  local metrics_json="${final_dir}/metrics_${metrics_tag}.json"
  local ops_ledger_json="${final_dir}/ops_ledger_${metrics_tag}.json"
  local src_final_dir="run_logs/final/${method}"
  local src_metrics="${src_final_dir}/metrics_${metrics_tag}.json"
  local src_ops_ledger="${src_final_dir}/ops_ledger_${metrics_tag}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "run_logs/wave1_final/${tag}.out" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="

  # -f guards: a crashed/incomplete run may not have produced these at all.
  [ -f "$src_metrics" ] && mv -f "$src_metrics" "$metrics_json"
  [ -f "$src_ops_ledger" ] && mv -f "$src_ops_ledger" "$ops_ledger_json"

  if [ "$method" = "sketchlora" ]; then
    # models/sketchlora.py::_diag_path, reproduced exactly (not re-derived by
    # hand in bash, to avoid drifting from the actual formula) -- run this
    # config's own dataset/split/energy_target/n_lora_blocks/seed through the
    # SAME logic that method uses to build its own output filename.
    local src_diag
    src_diag=$(python -c "
import json
c = json.load(open('${config}'))
eps = c.get('svd_energy_target')
nb = c.get('n_lora_blocks')
split = 'ic{}i{}'.format(c['init_cls'], c['increment'])
if eps is not None:
    t = '{}_adapt{}_b{}_{}'.format(c['dataset'], eps, nb or 'all', split)
else:
    t = '{}_r{}_b{}_{}'.format(c['dataset'], c['svd_rank'], nb or 'all', split)
seed = c['seed'][0] if isinstance(c['seed'], list) else c['seed']
print('run_logs/sketchlora_diag_{}_seed{}.json'.format(t, seed))
")
    if [ -f "$src_diag" ]; then
      mkdir -p "$diag_dir"
      mv -f "$src_diag" "${diag_dir}/$(basename "$src_diag")"
    fi
  fi
}

echo "==== PHASE 1 (small datasets) start $(date) ===="
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${SMALL_DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one "$METHOD" "$DATASET" "$SEED"
    done
  done
done
echo "==== PHASE 1 (small datasets) complete (or wall reached) $(date) ===="

echo "==== PHASE 2 (omnibenchmark1k) start $(date) ===="
for METHOD in "${METHODS[@]}"; do
  for DATASET in "${BIG_DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      run_one "$METHOD" "$DATASET" "$SEED"
    done
  done
done
echo "==== PHASE 2 (omnibenchmark1k) complete (or wall reached) $(date) ===="

echo "==== ALL WAVE1_FINAL RUNS COMPLETE (or wall reached) $(date) ===="
