#!/bin/bash
# One XGBoost tuning run on SLURM.
# Submitted by submit_xgboost_all.sh.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/tuning/%x_%j.log

set -eo pipefail

PYTHON=.venv/bin/python

KEY="$1"    # xgboost | xgboost_noweather
CITY="$2"   # seoul | london | washington

T=forecasting/models/tuning

case "$KEY" in
    xgboost)
        ARGS=("$T/tune_xgboost.py" --scenario clean_only)
        ;;
    xgboost_noweather)
        ARGS=("$T/tune_xgboost.py" --scenario no_weather)
        ;;
    *)
        echo "Unknown XGBoost key: $KEY"
        exit 2
        ;;
esac

case "$CITY" in
    seoul|london|washington) ;;
    *)
        echo "Unknown city: $CITY"
        exit 2
        ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-local} | $KEY | $CITY | $(hostname) | CPUs $OMP_NUM_THREADS | $(date '+%F %T')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$PYTHON" "${ARGS[@]}" --city "$CITY"

echo "Finished: $(date '+%F %T')"
