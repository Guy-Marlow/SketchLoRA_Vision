#!/bin/bash
#SBATCH --job-name=ce_smoke_imagenetr5t
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce_smoke_imagenetr5t/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce_smoke_imagenetr5t/slurm_%j.err
#
# CE-instrumentation smoke test (user request 2026-08-03; CORRECTED same day --
# an earlier version of this script ran bounded_memory boundary mode, which is
# NOT what this validates): 5 tasks of ImageNet-R (stop_after_tasks=5) trained
# with ORACLE (real task-boundary, plain incremental_train) boundaries, 1 seed
# (1993), all 5 methods (SeqLoRA, SketchLoRA, O-LoRA, InfLoRA, TreeLoRA) run in
# SEQUENCE on ONE GPU (per explicit user instruction -- no parallel FIFO-worker
# pattern here, unlike round2_slurm_grid.slurm/imagenetr_slurm_grid.slurm's
# 2-GPU shared-queue design, which ALSO doesn't apply here since those are both
# bounded_memory campaigns).
#
# This exercises trainer.py's oracle per-task loop, NOT models/bounded_memory_
# mixin.py -- the measured-CE wiring (docs/ce_profiling_implementation_plan.md)
# now lives in BOTH places (trainer.py got its own parallel CEProfileController/
# OpsLedger wiring added 2026-08-03 specifically because the oracle path never
# had ANY CE logging before, not even the pre-existing analytic-formula ledger).
# ce_profile_every=1 (baked into the configs, scripts/gen_ce_smoke_imagenetr5t_
# configs.py) profiles EVERY task -- this run IS the validation, so there is no
# reason to sample; note trainer.py's own caveat that oracle-mode profiling
# wraps the ENTIRE per-task incremental_train() call (all epochs), heavier than
# bounded_memory's epoch-0-only sampling, but fine at only 5 tasks.
#
# Resource request is per explicit user spec: 1 GPU, 10 CPU cores, 32GB RAM --
# deliberately much smaller than round2_slurm_grid's 2 GPU/18 cpu/128GB, since
# this is a single-GPU, 5-task-truncated smoke test, not a production grid cell.
#
# Config generation: scripts/gen_ce_smoke_imagenetr5t_configs.py, copies its
# per-method hyperparameters VERBATIM from scripts/gen_imagenetr_slurm_grid_
# configs.py's own METHOD_CFG (rank/batch/epoch/optimizer/regularizer settings),
# but WITHOUT boundary_mode/bm_budget_mb (oracle, not bounded_memory) and WITH
# final_metrics=true (required -- see that script's docstring: this single flag
# is what activates both MetricsLogger and the new CE ledger in trainer.py's
# oracle loop; every OTHER experiment track sharing trainer.py is unaffected
# since none of them set it).
#
# OUTPUT CONSOLIDATION (user request: "all of the logging from this smoke test
# should go to its own folder, which I can then transfer back here via SCP"):
# the harness's own output paths are NOT changed here -- still
# run_logs/final/<method>/{metrics,ops_ledger}_<tag>.json, exactly like every
# other final_metrics run (oracle mode has no separate accuracy-only json the
# way bounded_memory has boundedmem_*.json -- MetricsLogger's own metrics_*.json
# already carries the accuracy curve via record_task, and "status":"done" once
# finalize() runs). Instead, after each method's run finishes, this script
# COPIES (not moves -- originals stay in place, so the skip-if-done
# resumability check and every other campaign's file layout are undisturbed)
# the two files that run produced into run_logs/ce_smoke_imagenetr5t/, so a
# single `scp -r run_logs/ce_smoke_imagenetr5t` after the job finishes retrieves
# everything: 5x metrics_*.json (accuracy + runtime + persistent-memory + FLOPs
# -- a sanity cross-check, this smoke test is not a real accuracy result), 5x
# ops_ledger_*.json (the actual measured-CE data -- this is the file to inspect
# first), plus this script's own slurm_%j.out/.err and each run's own captured
# stdout/stderr.
#
# RESUMABLE / LOCKFILE: same convention as round2_slurm_grid.slurm -- skip a
# method whose metrics json already has "status": "done". Sequential, not a
# FIFO queue (only 1 GPU here), so this is a plain for-loop rather than
# round2_slurm_grid.slurm's two-worker shared-queue idiom.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/ce_smoke_imagenetr5t

LOCKFILE="run_logs/ce_smoke_imagenetr5t/.ce_smoke_imagenetr5t.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another ce_smoke_imagenetr5t.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Pre-flight sanity check (same convention as round2_slurm_grid.slurm).
if ! python -c "import torch, timm; assert torch.cuda.is_available()" 2>/tmp/ce_smoke_imagenetr5t_preflight_$$.err; then
  echo "FATAL: pre-flight check failed (torch/timm import or CUDA unavailable). Details:"
  cat /tmp/ce_smoke_imagenetr5t_preflight_$$.err
  rm -f /tmp/ce_smoke_imagenetr5t_preflight_$$.err
  exit 1
fi
rm -f /tmp/ce_smoke_imagenetr5t_preflight_$$.err

echo "==== config generation start $(date) ===="
python scripts/gen_ce_smoke_imagenetr5t_configs.py
echo "==== config generation done $(date) ===="

SEED=1993
METHODS=(seqlora sketchlora olora inflora treelora)

echo "==== ce_smoke_imagenetr5t start: ${#METHODS[@]} methods, oracle boundaries, 5 tasks, 1 seed (${SEED}), 1 GPU, sequential $(date) ===="

run_one() {
  # $1=method
  local method="$1"
  local tag="${method}_s${SEED}"
  local config="exps/ce_smoke_imagenetr5t/${tag}.json"
  local prefix="ce_smoke_imagenetr5t_${method}_s${SEED}"
  # matches trainer.py::_train's oracle-path MetricsLogger/OpsLedger tag exactly:
  # _tag = "{model_name}_{dataset}_{prefix}_s{seed}"
  local mlog_tag="${method}_imagenetr_${prefix}_s${SEED}"
  local metrics_json="run_logs/final/${method}/metrics_${mlog_tag}.json"
  local ledger_json="run_logs/final/${method}/ops_ledger_${mlog_tag}.json"

  if [ ! -f "$config" ]; then
    echo "==== SKIP (no config) ${tag} $(date) ===="
    return
  fi
  if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
    echo "==== SKIP (done) ${tag} $(date) ===="
  else
    echo "==== START ${tag} $(date) ===="
    python main.py --config "$config" \
      > "run_logs/ce_smoke_imagenetr5t/${tag}.out" 2>&1 || true
    echo "==== DONE  ${tag} $(date) ===="
  fi

  # Consolidate this run's output into the smoke-test folder for SCP (copy, not
  # move -- see script header). Missing files are logged, not fatal, so one
  # method's incomplete run doesn't stop the rest of the sequence.
  for f in "$metrics_json" "$ledger_json"; do
    if [ -f "$f" ]; then
      cp "$f" "run_logs/ce_smoke_imagenetr5t/"
    else
      echo "WARNING: expected output missing for ${tag}: $f"
    fi
  done
}

for METHOD in "${METHODS[@]}"; do
  run_one "$METHOD"
done

echo "==== ALL RUNS COMPLETE (or wall reached) $(date) ===="
echo "==== everything needed is under run_logs/ce_smoke_imagenetr5t/ -- scp -r that directory back ===="
