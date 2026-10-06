#!/bin/bash
# Submit the SARIMAX tuning runs (clean_only) to SLURM
# for Seoul, London, and Washington.
#
# Run from the repo root:
#   bash submit_sarimax_all.sh

set -eo pipefail
cd "$(dirname "$0")"

CITIES=(seoul london washington)

CPUS=4
MEM=16G
TIME=48:00:00

if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Uncommitted code changes in this copy of the repo:"
    git status --short --untracked-files=no
    echo "Commit on the laptop and copy the repo again (or run with FORCE=1)."
    exit 1
fi

mkdir -p logs/tuning results/tuning

for CITY in "${CITIES[@]}"; do
    sbatch --job-name="tune_sarimax_${CITY}" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
        sarimax_tune_job.sh "$CITY"
done

echo "Submitted all 3 SARIMAX tuning jobs."
echo "Check with: squeue -u $USER"
