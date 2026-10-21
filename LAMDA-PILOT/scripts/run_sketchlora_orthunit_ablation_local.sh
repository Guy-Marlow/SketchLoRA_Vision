#!/usr/bin/env bash
# Sequential local runner for the sketchlora_orthunit_ablation campaign (6
# runs: ImageNet-R-20t + CIFAR-100-10t, 3 seeds each) on this single-GPU
# A5000 machine. Tests align_mode="orth_unit" (L2-normalized row-direction
# orthogonality penalty, 2026-10-20 user-requested ablation) against the
# project's standing "orth" (raw, unnormalized) default -- identical
# hyperparams otherwise, copied verbatim from sketchlora_orth_wave1_datasets.
# One run at a time -- .out logs land under
# run_logs/sketchlora_orthunit_ablation/<dataset>/<prefix>.out.
set -e
cd "$(dirname "$0")/.."
PYTHON="/c/Users/gmar762/AppData/Local/anaconda3/envs/treelora/python.exe"

CONFIGS=(
  exps/sketchlora_orthunit_ablation/cifar224_s1993.json
  exps/sketchlora_orthunit_ablation/cifar224_s1996.json
  exps/sketchlora_orthunit_ablation/cifar224_s1999.json
  exps/sketchlora_orthunit_ablation/imagenetr_s1993.json
  exps/sketchlora_orthunit_ablation/imagenetr_s1996.json
  exps/sketchlora_orthunit_ablation/imagenetr_s1999.json
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .json)
  dataset=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset'])" "$cfg")
  outdir="run_logs/sketchlora_orthunit_ablation/${dataset}"
  mkdir -p "$outdir"
  outfile="${outdir}/${name}.out"
  echo "=== $(date) starting ${cfg} -> ${outfile} ==="
  "$PYTHON" main.py --config "$cfg" > "$outfile" 2>&1
  echo "=== $(date) finished ${cfg} (exit $?) ==="
done

echo "=== all sketchlora_orthunit_ablation runs complete ==="
