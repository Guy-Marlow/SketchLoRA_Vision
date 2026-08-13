#!/bin/bash
#SBATCH --job-name=treelora_rerun_rainbowprompt
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/treelora_rerun_and_rainbowprompt_wave1_final/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/treelora_rerun_and_rainbowprompt_wave1_final/slurm_%j.err
#
# 2026-08-12 user request, follow-up to wave1_final: two campaigns, folded
# into the SAME results family (exps/wave1_final/, run_logs/wave1_final/),
# not a standalone campaign directory.
#
# (a) TREELORA RERUN, same 15 configs (exps/wave1_final/treelora_{dataset}_
#     s{seed}.json, all 5 datasets x 3 seeds, UNCHANGED -- this campaign
#     generates no new TreeLoRA configs), under the CORRECTED code
#     (models/treelora.py, this session -- removed the per-task adapter bank
#     the reference implementation doesn't have; see the
#     treelora_adapter_bank_removed memory for the full finding and the
#     smoke-test A/B that validated it: [86.44,79.39,72.19] (old, broken
#     bank) -> [86.75,79.54,73.91] (new, faithful) on a 3-task ImageNet-R
#     probe). Every wave1_final TreeLoRA number on disk right now reflects
#     the OLD, invalid architecture -- this rerun exists specifically to
#     replace it, "same config, corrected code" was explicit in the request.
#
#     BACKUP-BEFORE-OVERWRITE (see backup_stale_treelora() below): the
#     standard resumable skip-check every script in this project uses (skip a
#     cell whose metrics json already reads "status": "done") would otherwise
#     skip EVERY TreeLoRA cell outright, since the old, invalid run already
#     completed and wrote exactly that. Before the TreeLoRA loop starts, this
#     script moves any existing run_logs/wave1_final/final/treelora/{metrics,
#     ops_ledger}_treelora_*.json and run_logs/wave1_final/treelora_*.out into
#     run_logs/wave1_final/treelora_prebankfix_backup/ (idempotent -- a cell
#     already backed up is left alone, so resubmitting this script doesn't
#     re-move a since-completed rerun's fresh data back into the backup dir).
#     Once backed up, the normal skip-check correctly sees no "done" metrics
#     for treelora and reruns every cell; a rerun's own output then lands
#     back in the original run_logs/wave1_final/final/treelora/ location,
#     replacing the (now-backed-up) stale numbers -- same location, corrected
#     data, matching the "rerun on the same config" framing.
#
# (b) NEW RAINBOWPROMPT RUN, all 5 wave1_final datasets x 3 seeds = 15 runs,
#     NEW to this family (wasn't part of the original 5-method wave1_final
#     grid). init_lr=1e-3 uniformly -- the value the cllora_ease_tuna_rainbow
#     lr_refine sweep confirmed cleanly across all 5 datasets as
#     RainbowPrompt's best LR (see that campaign's analysis, this session).
#     Full untruncated task counts (10/20/10/20/100), final_metrics=true,
#     print_forget=true -- same conventions as every other wave1_final
#     method, unlike the lr_refine sweep's own truncated/final_metrics=false
#     probe configs. Configs generated fresh below by
#     scripts/gen_treelora_rerun_and_rainbowprompt_wave1_final_configs.py
#     (idempotent, safe to re-run), written into exps/wave1_final/ itself
#     (rainbowprompt_{dataset}_s{seed}.json) -- an extension of that
#     directory, not a new one. Its metrics/ops_ledger land in a NEW
#     run_logs/wave1_final/final/rainbowprompt/ subdirectory -- no collision
#     with anything else in the family.
#
# SINGLE GPU BY CONSTRUCTION, same as wave1_final.slurm: one --gres=gpu:1,
# one strictly sequential loop, CUDA_VISIBLE_DEVICES pinned to 0.
#
# RUN ORDER: TreeLoRA rerun first, then RainbowPrompt (matches the order the
# user listed them in). WITHIN each method: small datasets (cifar224,
# imagenetr, food101, sun397) before omnibenchmark1k (100 tasks, by far the
# most expensive single dataset in this grid) -- same phase-boundary
# discipline as wave1_final.slurm, for the same reason: a wall-clock cutoff
# (plausible even at 72h, once TreeLoRA rerun + a fresh 15-run RainbowPrompt
# grid are both in scope) degrades gracefully to "small-dataset results
# secured for both methods" rather than risking a partial omnibenchmark1k for
# one method while the other's small-dataset cells haven't even started.
# 4 phases total: treelora-small (12) -> treelora-omni (3) ->
# rainbowprompt-small (12) -> rainbowprompt-omni (3) = 30 runs.
#
# RESUMABLE / LOCKFILE: same convention as every other script in this
# project -- skips a cell whose metrics json already shows "status": "done"
# (post-backup, for treelora); refuses to start a second instance of itself.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }
mkdir -p run_logs/treelora_rerun_and_rainbowprompt_wave1_final run_logs/wave1_final

LOCKFILE="run_logs/wave1_final/.treelora_rerun_and_rainbowprompt.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another treelora_rerun_and_rainbowprompt_wave1_final.slurm instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

source /data/gmar762/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME:-treelora}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

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

echo "==== RainbowPrompt config generation start $(date) ===="
python scripts/gen_treelora_rerun_and_rainbowprompt_wave1_final_configs.py
echo "==== RainbowPrompt config generation done $(date) ===="
echo "==== TreeLoRA configs: reused unchanged from exps/wave1_final/treelora_*.json (no generation needed) ===="

SEEDS=(1993 1996 1999)
SMALL_DATASETS=(cifar224 imagenetr food101 sun397)
BIG_DATASETS=(omnibenchmark1k)

backup_stale_treelora() {
  # One-time (idempotent) move of the pre-bankfix TreeLoRA results out of the
  # way so run_one()'s "status": "done" skip-check reruns every cell instead
  # of skipping all 15 as already complete. See header comment for the full
  # rationale.
  local backup_dir="run_logs/wave1_final/treelora_prebankfix_backup"
  local src_dir="run_logs/wave1_final/final/treelora"
  mkdir -p "$backup_dir"
  for dataset in "${SMALL_DATASETS[@]}" "${BIG_DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local tag="treelora_${dataset}_s${seed}"
      local prefix="wave1_final_${tag}"
      # BUG FIX (2026-08-13): was "${tag}_${prefix}_s${seed}" -- tag already
      # ends in _s${seed}, so that duplicated the seed and never matched the
      # real file (metrics_treelora_<dataset>_wave1_final_treelora_<dataset>_
      # s<seed>_s<seed>.json). Silently found nothing to back up on every
      # cell, every submission -- run_one()'s own (correct) metrics_tag below
      # then saw the untouched pre-fix "status": "done" file and skipped the
      # rerun entirely. Confirmed on a live H200 run: the "new" and original
      # wave1_final TreeLoRA metrics came back byte-for-byte identical.
      local metrics_tag="treelora_${dataset}_${prefix}_s${seed}"
      local metrics_json="${src_dir}/metrics_${metrics_tag}.json"
      local ops_ledger_json="${src_dir}/ops_ledger_${metrics_tag}.json"
      local out_log="run_logs/wave1_final/${tag}.out"
      # already backed up -> this cell's stale data was already moved aside
      # on an earlier submission; a metrics json present now is either a
      # completed rerun (leave it -- the skip-check should honor it) or a
      # not-yet-started/in-progress rerun (nothing to move). Either way,
      # skip.
      if [ -f "${backup_dir}/$(basename "$metrics_json")" ]; then
        continue
      fi
      if [ -f "$metrics_json" ] && grep -q '"status": "done"' "$metrics_json"; then
        echo "==== BACKUP (pre-bankfix) ${tag} $(date) ===="
        mv -f "$metrics_json" "${backup_dir}/"
        [ -f "$ops_ledger_json" ] && mv -f "$ops_ledger_json" "${backup_dir}/"
        [ -f "$out_log" ] && mv -f "$out_log" "${backup_dir}/$(basename "$out_log")"
      fi
    done
  done
}

run_one() {
  # $1=method $2=dataset $3=seed
  local method="$1" dataset="$2" seed="$3"
  local tag="${method}_${dataset}_s${seed}"
  local config="exps/wave1_final/${tag}.json"
  local prefix="wave1_final_${tag}"
  local metrics_tag="${method}_${dataset}_${prefix}_s${seed}"

  # CONSOLIDATION, same convention as wave1_final.slurm: trainer.py/
  # utils/metrics_logger.py/utils/ops_ledger.py write to their own shared
  # run_logs/final/<model_name>/ first, then get moved here.
  local final_dir="run_logs/wave1_final/final/${method}"
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

  [ -f "$src_metrics" ] && mv -f "$src_metrics" "$metrics_json"
  [ -f "$src_ops_ledger" ] && mv -f "$src_ops_ledger" "$ops_ledger_json"
}

echo "==== backing up pre-bankfix TreeLoRA results (idempotent) $(date) ===="
backup_stale_treelora
echo "==== backup step done $(date) ===="

echo "==== PHASE 1 (TreeLoRA rerun, small datasets) start $(date) ===="
for DATASET in "${SMALL_DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "treelora" "$DATASET" "$SEED"
  done
done
echo "==== PHASE 1 complete (or wall reached) $(date) ===="

echo "==== PHASE 2 (TreeLoRA rerun, omnibenchmark1k) start $(date) ===="
for DATASET in "${BIG_DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "treelora" "$DATASET" "$SEED"
  done
done
echo "==== PHASE 2 complete (or wall reached) $(date) ===="

echo "==== PHASE 3 (RainbowPrompt, small datasets) start $(date) ===="
for DATASET in "${SMALL_DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "rainbowprompt" "$DATASET" "$SEED"
  done
done
echo "==== PHASE 3 complete (or wall reached) $(date) ===="

echo "==== PHASE 4 (RainbowPrompt, omnibenchmark1k) start $(date) ===="
for DATASET in "${BIG_DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "rainbowprompt" "$DATASET" "$SEED"
  done
done
echo "==== PHASE 4 complete (or wall reached) $(date) ===="

echo "==== ALL TREELORA-RERUN + RAINBOWPROMPT RUNS COMPLETE (or wall reached) $(date) ===="
