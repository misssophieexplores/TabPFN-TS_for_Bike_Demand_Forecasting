#!/bin/bash
# Submit all XGBoost tuning runs to SLURM:
#   - clean_only
#   - no_weather
# for Seoul, London, and Washington.
#
# Run from the repo root:
#   bash submit_xgboost_all.sh

set -eo pipefail
cd "$(dirname "$0")"

CITIES=(seoul london washington)
KEYS=(xgboost xgboost_noweather)

CPUS=8
MEM=16G
TIME=48:00:00

if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Uncommitted code changes in this copy of the repo:"
    git status --short --untracked-files=no
    echo "Commit on the laptop and copy the repo again (or run with FORCE=1)."
    exit 1
fi

mkdir -p logs/tuning results/tuning

for KEY in "${KEYS[@]}"; do
    for CITY in "${CITIES[@]}"; do
        sbatch             --job-name="tune_${KEY}_${CITY}"             --cpus-per-task="$CPUS"             --mem="$MEM"             --time="$TIME"             xgboost_tune_job.sh "$KEY" "$CITY"
    done
done

echo "Submitted all 6 XGBoost tuning jobs."
echo "Check with: squeue -u $USER"
