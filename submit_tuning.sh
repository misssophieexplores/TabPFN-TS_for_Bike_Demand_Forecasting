#!/bin/bash
# Submit the v7 tuning runs to SLURM: one job per model and city.
# Run on the cluster, from the repo root:
#   bash submit_tuning.sh                        -> all 21 jobs
#   bash submit_tuning.sh xgboost_noweather      -> only the listed models
# Logs: logs/tuning/tune_<model>_<city>_<jobid>.log
# Results: results/tuning/*.json (then put the paths in the city configs)

set -eo pipefail
cd "$(dirname "$0")"   # repo root (this script lives there)

ALL_KEYS=(arima sarimax xgboost xgboost_noweather prophet neuralprophet neuralprophet_noweather)
CITIES=(seoul london washington)
TIME=48:00:00   # time limit per job
MEM=16G         # memory per job

if [ $# -gt 0 ]; then KEYS=("$@"); else KEYS=("${ALL_KEYS[@]}"); fi
for KEY in "${KEYS[@]}"; do
    if [[ ! " ${ALL_KEYS[*]} " =~ " ${KEY} " ]]; then
        echo "Unknown model key: $KEY. Available: ${ALL_KEYS[*]}"; exit 2
    fi
done

# Every tuning file records the git commit. Code changed on the cluster would
# not match any commit, so stop here.
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Uncommitted code changes in this copy of the repo:"
    git status --short --untracked-files=no
    echo "Commit on the laptop and copy the repo again (or run with FORCE=1)."
    exit 1
fi

mkdir -p logs/tuning results/tuning

for KEY in "${KEYS[@]}"; do
    case "$KEY" in
        xgboost|xgboost_noweather) CPUS=8 ;;   # XGBoost trains multi-threaded
        *)                         CPUS=4 ;;
    esac
    for CITY in "${CITIES[@]}"; do
        sbatch --job-name="tune_${KEY}_${CITY}" --cpus-per-task="$CPUS" \
               --mem="$MEM" --time="$TIME" tune_job.sh "$KEY" "$CITY"
    done
done

echo "Submitted. Check with: squeue -u $USER"
