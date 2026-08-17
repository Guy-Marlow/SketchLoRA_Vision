#!/bin/bash
#SBATCH --job-name=olora_mechanism_ablation
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/olora_mechanism_ablation/slurm_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/olora_mechanism_ablation/slurm_%j.err
#
# O-LoRA mechanism-isolation ablation, ImageNet-R-20t, oracle CIL, 3 seeds
# (1993/1996/1999), 2026-08-17. 21 runs total.
#
# QUESTION. In bankcap_wave1_imagenetr20t, O-LoRA held to a 3-4MB bank ends up
# with exactly TWO LoRA slots (verified from that campaign's own logs: the total
# param count grows by one slot at task 1 and by head-only thereafter; epoch-1
# Orth stays flat at ~5.1 for tasks 2-19 instead of growing linearly with slot
# count as it does uncapped) -- yet it still scores 77.30 top5 against SeqLoRA's
# 65.66 at essentially the same footprint (2 slots + head = 3.40MB counted,
# ~4.10MB once O-LoRA's own uncounted _orth_prev_{q,v} cache is included, vs
# SketchLoRA's 4.00MB budget). Two things differ between "O-LoRA capped to 2
# slots" and SeqLoRA, and that campaign cannot separate them:
#
#   D1 (additive memory)  the frozen slot 0's delta is summed into every
#                         forward (train_merge=True), so the deployed function
#                         permanently carries a task-0-specialised component.
#   D2 (regulariser)      the live slot's A is pushed orthogonal to the frozen
#                         A_0 every step at lamda_1=0.5 -- a term worth ~2.5 of
#                         loss at epoch 1 against a CE of ~1.5-2.0, decaying to
#                         ~0.02 by epoch 20, every task.
#
# The existing evidence favours D2: in the final per-class-block accuracies the
# retention gain over SeqLoRA is spread UNIFORMLY across all 19 old tasks
# (32-67% vs SeqLoRA's 5-36%) at unchanged new-task accuracy (84.5 vs 85.2),
# whereas slot 0 only ever encoded task 0 -- an additive-memory effect would
# show up as a spike in the 00-09 block alone. If that reading is right, then
# O-LoRA's dominant continual-learning mechanism on this benchmark is a
# CONSTANT-MEMORY regulariser needing exactly one frozen reference slot, the
# bank cap cannot neutralise it, and the whole bankcap_wave1 table has to be
# read as measuring O-LoRA's secondary additive-capacity term only.
#
# THREE ARMS.
#
#   Arm A "noorth" -- D1 WITHOUT D2. O-LoRA at the same three derived caps
#     (3/4/6MB = 2/2/4 slots) with lamda_1=0.0. Config-only. 9 runs.
#
#   Arm B "nomerge" -- D2 WITHOUT D1, and the clean one. Same caps, lamda_1
#     untouched at 0.5, but lora_train_merge=false + lora_merge=false, so the
#     frozen slots stop contributing to the forward while still anchoring the
#     orthogonality penalty. The penalty is then NUMERICALLY IDENTICAL to the
#     capped run's (same fresh-init live slot, same frozen A_0 reference, same
#     _orth_prev cache) -- only the additive term is removed. Needed a one-line
#     change in models/olora.py to make train_merge configurable; default stays
#     True, so every pre-existing O-LoRA config is byte-identical. 9 runs.
#
#   Arm C "seqloraorth" -- the constant-memory question. models/seqlora_orth.py:
#     SeqLoRA's single pinned slot, never re-initialised, never merged with
#     anything, plus the same penalty against a frozen snapshot of that slot's
#     OWN task-0 A. Total footprint 1 slot + reference + head = 2.70MB, CONSTANT
#     in task count, below every cap above. No bank_cap_mb (there is no bank).
#     3 runs.
#
#     NOT a drop-in substitute for Arm B, and the smoke test showed why: because
#     the reference is a snapshot of the same matrix that keeps training, the
#     penalty starts each post-task-0 run at the self-Gram value |A_0 A_0^T|
#     (measured: 82.6 at task 1, 1 epoch) instead of the near-orthogonal
#     fresh-init value O-LoRA sees (7.06). Same formula and same lamda_1, but a
#     substantially HARSHER constraint -- it must push A off its own task-0
#     subspace, not merely off a different slot's. Read Arm C as "can a single
#     constant-memory slot get the benefit at all", not as a matched replicate
#     of Arm B. If Arm C underperforms, the harsher initial condition is a live
#     alternative explanation and lamda_1 would want a sweep before concluding
#     anything.
#
# READING THE RESULT (against bankcap's O-LoRA 77.30 / SeqLoRA 65.66 top5):
#   A ~ SeqLoRA and B ~ capped O-LoRA -> D2 confirmed. The cap does not bind on
#     O-LoRA's real mechanism and "O-LoRA at SketchLoRA's memory" is not
#     measuring what the campaign framing claims.
#   A ~ capped O-LoRA and B ~ SeqLoRA -> D1. The frozen additive slot was doing
#     the work and the uniform-retention argument above is wrong.
#   Both intermediate -> the two mechanisms compose; report both terms.
#   C tracks B -> the benefit is available at 2.70MB constant, which is the
#     directly SketchLoRA-relevant statement.
#
# WALL TIME: 21 runs x ~28-30 min/run (the rate bankcap_wave1 actually measured
# for O-LoRA at 20 tasks / 20 epochs, uniform across methods) = ~10.5h. 12h
# requested (2026-08-17 user decision, down from 14h) -- that is ~1.5h of
# headroom, i.e. tolerance for ~3 runs going long, not for a systematically
# slower node. Arm B's runs are the cheapest of the three (merge=false means
# the forward sums one slot instead of two, so if anything the ~28-30min/run
# figure is conservative for 9 of the 21). If the job does hit the wall,
# nothing is lost: resubmit and the finished runs skip in minutes (see below).
# Resumable: run_one() skips any config whose
# final_metrics JSON already reports status=="done", same check
# (scripts/_bankcap_run_done.py) and same rationale as bankcap_wave1 -- see that
# script's comment for why the .out log's "Average Accuracy (CNN):" line is NOT
# a valid completion signal.
#
# Configs are STATIC here (committed under exps/olora_mechanism_ablation/),
# unlike bankcap_wave1's generator -- nothing about them is derived at runtime,
# the caps are the three constants that campaign already resolved.
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=olora_mechanism_ablation
CONFIG_ROOT="exps/$CAMPAIGN"
OUT_ROOT="run_logs/$CAMPAIGN"
mkdir -p "$OUT_ROOT"

if [ ! -d "$CONFIG_ROOT" ]; then
  echo "FATAL: $CONFIG_ROOT not found -- the 21 configs are committed to the repo, not generated. Pull the branch that adds them."
  exit 1
fi

LOCKFILE="$OUT_ROOT/.$CAMPAIGN.lock"
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
echo "==== pre-flight OK, CUDA_VISIBLE_DEVICES=0 $(date) ===="

run_one() {
  local config="$1"
  local tag
  tag="$(basename "$config" .json)"
  local outfile="${OUT_ROOT}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${tag} $(date) ===="
    return
  fi
  echo "==== START ${tag} $(date) ===="
  python main.py --config "$config" \
    > "$outfile" 2>&1 || true
  echo "==== DONE  ${tag} $(date) ===="
}

echo "==== ${CAMPAIGN} start: arm A (9) + arm B (9) + arm C (3) = 21 runs $(date) ===="

echo "-------- ARM A: capped O-LoRA, orthogonality penalty OFF (D1 only) --------"
for cap in 3 4 6; do
  for cfg in "$CONFIG_ROOT"/olora_noorth_cap${cap}_s*.json; do
    run_one "$cfg"
  done
done

echo "-------- ARM B: capped O-LoRA, frozen slots out of the forward (D2 only) --------"
for cap in 3 4 6; do
  for cfg in "$CONFIG_ROOT"/olora_nomerge_cap${cap}_s*.json; do
    run_one "$cfg"
  done
done

echo "-------- ARM C: SeqLoRA + frozen-task-0 penalty, 2.70MB constant --------"
for cfg in "$CONFIG_ROOT"/seqlora_orth_s*.json; do
  run_one "$cfg"
done

echo "==== ALL ${CAMPAIGN} RUNS COMPLETE $(date) ===="
