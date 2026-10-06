#!/bin/bash
# One SARIMAX tuning run (clean_only) on SLURM.
# Submitted by submit_sarimax_all.sh.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/tuning/%x_%j.log

set -eo pipefail

PYTHON=.venv/bin/python

CITY="$1"   # seoul | london | washington

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

# Stop now if the params file cannot be written (otherwise this fails only at the end, after the tuning)
PROBE="results/tuning/.write_test_${SLURM_JOB_ID:-local}"
if ! { mkdir -p results/tuning && touch "$PROBE" && rm -f "$PROBE"; } 2>/dev/null; then
    echo "ERROR: results/tuning is not writable on $(hostname). Nothing was tuned."
    exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} | sarimax clean_only | $CITY | $(hostname) | CPUs $OMP_NUM_THREADS | $(date '+%F %T')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$PYTHON" forecasting/models/tuning/tune_sarimax.py --scenario clean_only --city "$CITY"

echo "Finished: $(date '+%F %T')"
echo "New params file (set as sarimax_params_file in forecasting/config_${CITY}.py):"
ls -t results/tuning/sarimax_best_params_"${CITY}"_clean_only_720_*.json 2>/dev/null | head -1 \
    || echo "  no file matching results/tuning/sarimax_best_params_${CITY}_clean_only_720_*.json"
