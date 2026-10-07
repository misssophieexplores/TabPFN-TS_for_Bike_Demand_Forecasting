#!/bin/bash
# One part of the v7 experiment run: python forecasting/main.py
# Submitted by submit_main.sh (sets job name, CPUs, memory, time limit and
# MAIN_RUN, MAIN_PART, MAIN_PARTS, MAIN_FRESH).
#
# Part 1 first:
#   - FRESH_START=1 (MAIN_FRESH=1): moves the results files of this
#     results_version into results/archive_<run>/, so everything is rerun with
#     the current code (one code ID in all results);
#   - writes the re-selected SARIMAX params files (converged candidates only)
#     if they do not exist yet.
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

# SARIMAX tuning files (7 Oct 2026) -> re-selected params files named in the city configs
SARIMAX_RESELECT=(
    "seoul|sarimax_best_params_seoul_clean_only_720_20261007_063501.json"
    "london|sarimax_best_params_london_clean_only_720_20261007_045557.json"
    "washington|sarimax_best_params_washington_clean_only_720_20261007_043638.json"
)

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

# Main process: XGBoost and torch use all CPUs of the job. SARIMAX folds run
# in one worker process per CPU, each with one math thread (fold_runner.py).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

reachable() {  # host: TCP connection to port 443 within 10 s
    timeout 10 bash -c "exec 3<>/dev/tcp/$1/443" 2>/dev/null
}
PROXY="${HTTPS_PROXY:-}${https_proxy:-}"

# W&B online if the node reaches W&B and an API key is set up; offline
# otherwise (wandb.init would fail and main.py would skip the city).
# Offline runs are kept in wandb/ and can be uploaded later with `wandb sync`.
if [ -n "${WANDB_MODE:-}" ]; then
    WANDB_WHY="WANDB_MODE was already set"
elif ! { [ -n "$PROXY" ] || reachable api.wandb.ai; }; then
    export WANDB_MODE=offline; WANDB_WHY="api.wandb.ai not reachable from $(hostname)"
elif ! { [ -n "${WANDB_API_KEY:-}" ] || grep -qs "api.wandb.ai" ~/.netrc || grep -qs "^ *WANDB_API_KEY" .env forecasting/.env; }; then
    export WANDB_MODE=offline; WANDB_WHY="no W&B API key (WANDB_API_KEY, ~/.netrc or .env)"
else
    WANDB_WHY="api.wandb.ai reachable, API key found"
fi
# TimesFM weights: use the local Hugging Face cache when huggingface.co is not reachable
if [ -z "$PROXY" ] && ! reachable huggingface.co; then
    export HF_HUB_OFFLINE=1
fi
echo "W&B: ${WANDB_MODE:-online} ($WANDB_WHY) | HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"

PROBE="results/.write_test_${JOB}"
if ! { mkdir -p results/tuning && touch "$PROBE" && rm -f "$PROBE"; } 2>/dev/null; then
    fail "results/ is not writable on $(hostname) (job $JOB). main.py was not started."
fi

if [ "$PART" = 1 ]; then
    VERSION="$("$PYTHON" -c "import sys; sys.path.insert(0, 'forecasting'); from config import ForecastConfig; print(ForecastConfig().results_version)")" \
        || fail "could not read results_version from forecasting/config.py"
    OLD=()
    for f in results/checkpoint_*_"$VERSION".json results/results_master_"$VERSION".csv \
             results/detailed_results_master_"$VERSION".csv results/forecasts_*_"$VERSION".csv \
             results/comparative_metrics_"$VERSION".csv results/errors_"$VERSION".log; do
        [ -f "$f" ] && OLD+=("$f")
    done
    if [ "${MAIN_FRESH:-0}" = 1 ] && [ ${#OLD[@]} -gt 0 ]; then
        mkdir -p "results/archive_${RUN}" && mv "${OLD[@]}" "results/archive_${RUN}/" \
            || fail "could not move the old results files into results/archive_${RUN}/"
        echo "Fresh start: moved ${#OLD[@]} results files of version $VERSION into results/archive_${RUN}/"
    elif grep -qs '"SARIMAX"' results/checkpoint_*_"$VERSION".json; then
        fail "results/checkpoint_*_$VERSION.json lists SARIMAX experiments from before the re-selection. Submit with FRESH_START=1 (moves the old results into results/archive_<run>/)."
    fi

    for entry in "${SARIMAX_RESELECT[@]}"; do
        CITY="${entry%%|*}"; SRC="results/tuning/${entry#*|}"
        DST="results/tuning/sarimax_best_params_${CITY}_clean_only_720_converged.json"
        [ -f "$DST" ] && continue
        [ -f "$SRC" ] || fail "SARIMAX tuning file not found: $SRC"
        echo "===== SARIMAX re-selection: $CITY ====="
        RLOG="$STATE/reselect_sarimax_${CITY}.log"   # complete output incl. the params JSON
        if ! "$PYTHON" forecasting/models/tuning/reselect_sarimax.py --source "$SRC" --output "$DST" > "$RLOG" 2>&1; then
            cat "$RLOG"
            fail "SARIMAX re-selection failed for $CITY (output above, also in $RLOG)"
        fi
        grep -v "^ \|^[{}]\|PARAMS JSON\|^$" "$RLOG"
        [ -f "$DST" ] || fail "SARIMAX re-selection wrote no file for $CITY (see $RLOG)"
    done
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
