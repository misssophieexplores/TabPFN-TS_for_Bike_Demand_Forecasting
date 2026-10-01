#!/bin/bash
# One v7 tuning run on SLURM: one model, one city.
# Submitted by submit_tuning.sh (sets job name, CPUs, memory and time limit).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/tuning/%x_%j.log

set -eo pipefail

PYTHON=.venv/bin/python   # project environment, relative to the repo root

KEY="$1"    # arima | sarimax | xgboost | xgboost_noweather | prophet | neuralprophet | neuralprophet_noweather
CITY="$2"   # seoul | london | washington

T=forecasting/models/tuning
case "$KEY" in
    arima)                   ARGS=("$T/tune_arima.py") ;;
    sarimax)                 ARGS=("$T/tune_sarimax.py" --scenario clean_only) ;;
    xgboost)                 ARGS=("$T/tune_xgboost.py" --scenario clean_only) ;;
    xgboost_noweather)       ARGS=("$T/tune_xgboost.py" --scenario no_weather) ;;
    prophet)                 ARGS=("$T/tune_prophet.py") ;;
    neuralprophet)           ARGS=("$T/tune_neuralprophet.py" --scenario clean_only) ;;
    neuralprophet_noweather) ARGS=("$T/tune_neuralprophet.py" --scenario no_weather) ;;
    *) echo "Unknown model key: $KEY"; exit 2 ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # repo root (submit_tuning.sh submits from there)

# Keep numeric libraries (XGBoost, numpy, torch) within the CPUs SLURM gave this job
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-local} | $KEY | $CITY | $(hostname) | CPUs $OMP_NUM_THREADS | $(date '+%F %T')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
"$PYTHON" "${ARGS[@]}" --city "$CITY"
echo "Finished: $(date '+%F %T')"
