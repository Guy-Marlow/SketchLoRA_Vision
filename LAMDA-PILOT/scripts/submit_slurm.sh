#!/bin/bash
# Wrapper for `sbatch` that guarantees a campaign's run_logs/<campaign>/
# directory exists BEFORE sbatch parses the script's own #SBATCH --output/
# --error directives.
#
# WHY THIS HAS TO BE A SEPARATE WRAPPER, NOT SOMETHING THE .slurm SCRIPT DOES
# ITSELF (2026-08-19, after job 14891 died in 2s with no run_logs/
# sketchlora_fixedrank_lr_tune/ directory ever created, confirmed via
# `sacct`): SLURM opens --output/--error as part of LAUNCHING the job --
# before a single line of the script's own shell body executes, including
# any mkdir that script puts at its own top. This is not a SLURM quirk, it's
# how process I/O redirection works everywhere (the same reason
# `some_command > /no/such/dir/file` fails immediately in plain bash) -- the
# redirect target has to exist before the command starts, full stop. And
# run_logs/ is gitignored project-wide, so a freshly-cloned/pulled checkout
# never has any run_logs/<campaign>/ subdirectory at all -- every prior
# campaign's script "worked" only because SOME earlier run (or a manual
# mkdir) happened to have already created it. The only point in the whole
# pipeline that runs early enough to fix this is right here, before sbatch is
# even invoked.
#
# USAGE: scripts/submit_slurm.sh scripts/<campaign>.slurm [extra sbatch args...]
# Infers the campaign name from the script's own filename (matches every
# script's own CAMPAIGN=... variable / #SBATCH --job-name convention: a
# script named foo.slurm always logs to run_logs/foo/), mkdir -p's
# run_logs/<campaign>/, then hands off to sbatch unchanged.
#
# This is now the STANDARD way to submit any SLURM script in this project --
# do not call `sbatch scripts/foo.slurm` directly.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 scripts/<campaign>.slurm [extra sbatch args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift
if [ ! -f "$SCRIPT" ]; then
  echo "FATAL: $SCRIPT not found" >&2
  exit 1
fi

CAMPAIGN="$(basename "$SCRIPT" .slurm)"
VISION=/data/gmar762/research/continuous_learning/svd_sketching_vision/LAMDA-PILOT
mkdir -p "$VISION/run_logs/$CAMPAIGN"
echo "Created/confirmed $VISION/run_logs/$CAMPAIGN -- submitting $SCRIPT"
exec sbatch "$SCRIPT" "$@"
