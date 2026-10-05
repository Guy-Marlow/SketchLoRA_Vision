#!/usr/bin/env bash
# Sequential local runner for the freqdir_dense_wave1_v2 campaign (6 runs:
# ImageNet-R-20t + CIFAR-100-10t, 3 seeds each) on this single-GPU A5000
# machine. One run at a time -- .out logs land under run_logs/
# freqdir_dense_wave1_v2/<dataset>/<prefix>.out.
set -e
cd "$(dirname "$0")/.."
PYTHON="/c/Users/gmar762/AppData/Local/anaconda3/envs/treelora/python.exe"

CONFIGS=(
  exps/freqdir_dense_wave1_v2/cifar224_s1993.json
  exps/freqdir_dense_wave1_v2/cifar224_s1996.json
  exps/freqdir_dense_wave1_v2/cifar224_s1999.json
  exps/freqdir_dense_wave1_v2/imagenetr_s1993.json
  exps/freqdir_dense_wave1_v2/imagenetr_s1996.json
  exps/freqdir_dense_wave1_v2/imagenetr_s1999.json
)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .json)
  dataset=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset'])" "$cfg")
  outdir="run_logs/freqdir_dense_wave1_v2/${dataset}"
  mkdir -p "$outdir"
  outfile="${outdir}/${name}.out"
  echo "=== $(date) starting ${cfg} -> ${outfile} ==="
  "$PYTHON" main.py --config "$cfg" > "$outfile" 2>&1
  echo "=== $(date) finished ${cfg} (exit $?) ==="
done

echo "=== all freqdir_dense_wave1_v2 runs complete ==="
