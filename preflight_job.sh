#!/bin/bash
# Pre-flight check before main.py. Writes nothing to results/.
#
# Run from the repo root:
#   mkdir -p logs && sbatch preflight_job.sh                  -> full check (params files, build_models,
#                                                                one fold per model incl. TabPFN/TimesFM, W&B)
#   mkdir -p logs && sbatch preflight_job.sh --skip-models    -> quick check (configs and params files only)
#
# Log: logs/preflight_<jobid>.log

#SBATCH --job-name=preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.log

set -o pipefail

PYTHON=.venv/bin/python

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-local} | preflight $* | $(hostname) | CPUs $OMP_NUM_THREADS | $(date '+%F %T')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$PYTHON" forecasting/testing/preflight.py "$@"
RC=$?

if [ "$RC" -eq 0 ]; then
    echo "PREFLIGHT PASSED: $(date '+%F %T')"
else
    echo "PREFLIGHT FAILED (exit $RC): see the [FAIL] lines above. $(date '+%F %T')"
fi
exit "$RC"
