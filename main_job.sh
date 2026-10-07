#!/bin/bash
# One part of the v7 experiment run: python forecasting/main.py
# Submitted by submit_main.sh (sets job name, CPUs, memory, time limit, MAIN_RUN, MAIN_PART, MAIN_PARTS).
#
# - If main.py fails, all later parts exit at once.
# - 10 min before the time limit, main.py is stopped; the next part resumes from the checkpoints.
# - Once main.py has finished, all later parts exit at once.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/main/%x_%j.log

set -o pipefail

PYTHON=.venv/bin/python
RUN="${MAIN_RUN:?Submit with: bash submit_main.sh}"
PART="${MAIN_PART:-1}"
PARTS="${MAIN_PARTS:-1}"
JOB="${SLURM_JOB_ID:-local}"
LOG="logs/main/${SLURM_JOB_NAME:-main_v7}_${JOB}.log"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
STATE="logs/main/run_${RUN}"
mkdir -p "$STATE"

echo "Job $JOB | run $RUN | part $PART/$PARTS | $(hostname) | CPUs ${SLURM_CPUS_PER_TASK:-1} | $(date '+%F %T')"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [ -f "$STATE/DONE" ]; then
    echo "main.py already finished in an earlier part. Nothing to do."
    exit 0
fi
if [ -f "$STATE/FAILED" ]; then
    echo "Not started: an earlier part failed:"
    cat "$STATE/FAILED"
    exit 0
fi

fail() {
    echo "FAILED: $1" | tee "$STATE/FAILED"
    exit 1
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

# W&B offline: wandb.init cannot fail on a node without internet (runs are kept in wandb/)
export WANDB_MODE=offline
# TimesFM weights: use the local Hugging Face cache when huggingface.co is not reachable
if [ -z "${HTTPS_PROXY:-}${https_proxy:-}" ] && ! timeout 10 bash -c 'exec 3<>/dev/tcp/huggingface.co/443' 2>/dev/null; then
    export HF_HUB_OFFLINE=1
fi
echo "WANDB_MODE=$WANDB_MODE | HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"

PROBE="results/.write_test_${JOB}"
if ! { mkdir -p results && touch "$PROBE" && rm -f "$PROBE"; } 2>/dev/null; then
    fail "results/ is not writable on $(hostname) (job $JOB). main.py was not started."
fi

echo "===== main.py $(date '+%F %T') ====="
PID=""
STOPPING=0
on_time_limit() {
    STOPPING=1
    echo "Time limit in 10 min ($(date '+%F %T')): stopping main.py. Part $((PART + 1)) resumes from the checkpoints."
    if [ -n "$PID" ]; then
        pkill -KILL -P "$PID" 2>/dev/null
        kill -KILL "$PID" 2>/dev/null
    fi
}
trap on_time_limit USR1

"$PYTHON" forecasting/main.py &
PID=$!
wait "$PID"
RC=$?

if [ "$STOPPING" = 1 ]; then
    wait "$PID" 2>/dev/null
    if [ "$PART" -ge "$PARTS" ]; then
        echo "This was the last part ($PART/$PARTS) and main.py has not finished. Run 'bash submit_main.sh' again: it resumes from the checkpoints."
    fi
    exit 0
fi

if [ "$RC" -eq 0 ]; then
    echo "main.py finished: $(date '+%F %T')" | tee "$STATE/DONE"
    exit 0
fi
fail "main.py exit $RC (job $JOB): see $LOG. Later parts will not run."
