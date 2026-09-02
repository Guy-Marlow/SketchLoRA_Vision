#!/bin/bash
#SBATCH --job-name=ce_final_all_retry3
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --open-mode=append
#SBATCH --output=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce_final_all/slurm_retry3_%j.out
#SBATCH --error=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT/run_logs/ce_final_all/slurm_retry3_%j.err
#
# Targeted re-run of ce_final_all's 3 broken results (2026-09-02 fix) --
# NOT a resubmission of the full 9-method ce_final_all.slurm.
#
# From job 22626's logs (pulled locally under ce_final_all/):
#   - cllora:         OOM-killed mid-task-5 (host RAM, not CUDA -- confirmed
#                      via slurm_22626.err's "Detected 6 oom_kill events",
#                      no Python traceback).
#   - rainbowprompt:   OOM-killed mid-task-1, took the whole job down with it
#                      (it ran LAST in ce_final_all.slurm's METHOD_ORDER).
#   - sketchlora:      completed all 20 tasks with NO error, but trained the
#                      WRONG model -- scripts/gen_ce_final_all_configs.py's
#                      SOURCES["sketchlora"] entry (at the time job 22626's
#                      configs were generated) pointed at the base/non-orth
#                      config instead of the orth default (see
#                      scripts/ce_final_all_sketchlora_fix.slurm's docstring
#                      for the full history of this bug). The generator is
#                      already fixed as of 2026-09-01 (SOURCES["sketchlora"]
#                      = exps/sketchlora_fixedrank_orth05_imagenetr_s1993.json,
#                      model_name="sketchlora_align") -- config_generation
#                      below just needs to run again to pick that up.
#
# OOM ROOT CAUSE, user's working theory (2026-09-02): --mem was already
# raised once for this exact job type (64G -> 128G, see ce_final_all.slurm's
# own header) and STILL wasn't enough for cllora/rainbowprompt. Per-method
# architecture doesn't explain it either -- TUNA (heaviest by parameter/
# module count, by a wide margin) ran clean at position 7 of 9, while
# smaller cllora (position 5) and rainbowprompt (position 9, i.e. deepest
# into the sequential loop) both OOM'd. This fits SLURM/cgroup memory
# accounting (page cache and similar counted against the job's cgroup
# memory.limit_in_bytes, cgroup v1 default) accumulating across NINE
# sequential `python main.py` invocations sharing one long-lived job/cgroup,
# rather than any single method's own footprint being the problem --
# non-monotonic pressure (an OOM-kill itself forces a reclaim, which is
# consistent with tuna succeeding mid-sequence after cllora's crash freed
# things back up, only for pressure to build again by rainbowprompt at the
# tail end). NOT re-verified against actual `sacct -j 22626 --format=MaxRSS`
# peak-RSS data (no cluster shell access from the session that wrote this
# script) -- if this retry OOMs again under a FRESH job/cgroup with only 3
# (lighter, non-adjacent-in-the-old-order) methods, that would be strong
# evidence against the cross-run-accumulation theory and for a real
# per-method leak instead, and --mem would need raising for real next time.
#
# Kept at --mem=128G deliberately (not bumped) -- the theory says the
# ceiling itself was never the problem, just how deep into one shared job
# these methods happened to run. A fresh job gives cllora and rainbowprompt
# the full 128G to themselves instead of whatever was left after 4-8 prior
# methods' accumulated cgroup pressure.
#
# Same conventions as ce_final_all.slurm: CE1 forbidden (ce2_enabled=true
# unconditionally), resumable via scripts/_bankcap_run_done.py, existing
# outfile backed up before a retry overwrites it, CE2 JSON relocated via
# scripts/_move_ce2_output.py (sketchlora uses the dest_name override so its
# corrected sketchlora_align-keyed output lands as/overwrites ce2/sketchlora/
# rather than creating a separate ce2/sketchlora_align/ -- see that script's
# own docstring). ALL output stays under run_logs/ce_final_all/ so the same
# `scp -r run_logs/ce_final_all` pulls this retry's results too.
#
# Order: rainbowprompt first (it's the one that failed fastest and earliest
# last time, so it gets the freshest/most headroom), then cllora, then
# sketchlora last (no memory risk at all, just needs the corrected config).
#
# >>> CLUSTER-SPECIFIC: conda env name -- adjust CONDA_ENV_NAME if not "treelora" <<<

set -uo pipefail

VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
cd "$VISION" || { echo "FATAL: cannot cd to $VISION -- check the path for this cluster"; exit 1; }

CAMPAIGN=ce_final_all
CONFIG_DIR="exps/$CAMPAIGN"
OUT_DIR="run_logs/$CAMPAIGN"
CE2_DEST="$OUT_DIR/ce2"
mkdir -p "$CONFIG_DIR" "$OUT_DIR" "$CE2_DEST"

LOCKFILE="$OUT_DIR/.$CAMPAIGN.lock"
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
  echo "FATAL: another $CAMPAIGN.slurm (or a retry script) instance (PID $(cat "$LOCKFILE")) appears to be running -- refusing to start a second one. Delete $LOCKFILE if this is stale."
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

echo "==== config generation start (picks up the sketchlora orth fix) $(date) ===="
python scripts/gen_ce_final_all_configs.py
echo "==== config generation done $(date) ===="

run_one() {
  # $1 = path to config json. $2 = optional _move_ce2_output.py dest_name override.
  local config="$1"
  local dest_name_override="${2:-}"
  local tag
  tag="$(basename "$config" .json)"
  local outfile="${OUT_DIR}/${tag}.out"

  if python scripts/_bankcap_run_done.py "$config" 2>/dev/null; then
    echo "==== SKIP (done) ${tag} $(date) ===="
  else
    if [ -f "$outfile" ]; then
      mv "$outfile" "${outfile}.prev-${SLURM_JOB_ID:-unknown}"
    fi
    echo "==== START ${tag} $(date) ===="
    python main.py --config "$config" \
      > "$outfile" 2>&1 || true
    echo "==== DONE  ${tag} $(date) ===="
  fi
  if [ -n "$dest_name_override" ]; then
    python scripts/_move_ce2_output.py "$config" "$CE2_DEST" "$dest_name_override" 2>/dev/null || true
  else
    python scripts/_move_ce2_output.py "$config" "$CE2_DEST" 2>/dev/null || true
  fi
}

echo "==== ${CAMPAIGN}_retry3 start: rainbowprompt, cllora, sketchlora x 1 seed, single GPU $(date) ===="

run_one "$CONFIG_DIR/rainbowprompt_s1993.json"
run_one "$CONFIG_DIR/cllora_s1993.json"
run_one "$CONFIG_DIR/sketchlora_s1993.json" "sketchlora"

echo "==== ${CAMPAIGN}_retry3 COMPLETE (or wall reached) $(date) ===="
