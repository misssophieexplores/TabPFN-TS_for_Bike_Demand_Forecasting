#!/bin/bash
# Submit the v7 experiments (python forecasting/main.py) to SLURM.
#
# Run from the repo root:
#   bash submit_main.sh                  -> resumes from the checkpoints
#   FRESH_START=1 bash submit_main.sh    -> part 1 first moves the results files of this
#                                           results_version into results/archive_<run>/
#
# Submits PARTS jobs of 48 h each; each one starts when the previous one has ended.
#   - If main.py fails, the remaining parts exit within seconds.
#   - 10 min before a part's time limit, main.py is stopped and the next part
#     resumes from the checkpoints.
#   - Once main.py has finished, the remaining parts exit within seconds.
# Logs: logs/main/main_v7_<jobid>.log
# Stop everything: scancel -n main_v7

set -eo pipefail
cd "$(dirname "$0")"

PARTS=8          # 8 x 48 h = 16 days at most
CPUS=16          # SARIMAX fits 16 folds at a time
MEM=32G
TIME=48:00:00

if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "Uncommitted code changes in this copy of the repo:"
    git status --short --untracked-files=no
    echo "Commit on the laptop and copy the repo again (or run with FORCE=1)."
    exit 1
fi

if [ -n "$(squeue -u "$USER" -h -n main_v7 -o %i 2>/dev/null || true)" ]; then
    echo "main_v7 jobs are already queued or running:"
    squeue -u "$USER" -n main_v7
    echo "Two runs at once would write into the same results files. Nothing submitted."
    exit 1
fi

RUN="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs/main results

PREV=""
for PART in $(seq 1 "$PARTS"); do
    DEP=()
    if [ -n "$PREV" ]; then
        DEP=(--dependency="afterany:$PREV")
    fi
    OUT="$(sbatch --parsable "${DEP[@]}" --job-name=main_v7 --cpus-per-task="$CPUS" --mem="$MEM" \
           --time="$TIME" --export="ALL,MAIN_RUN=$RUN,MAIN_PART=$PART,MAIN_PARTS=$PARTS,MAIN_FRESH=${FRESH_START:-0}" \
           main_job.sh)"
    PREV="${OUT%%;*}"
    echo "Part $PART/$PARTS: job $PREV"
done

echo "Submitted run $RUN: $PARTS parts of $TIME, $CPUS CPUs.$([ "${FRESH_START:-0}" = 1 ] && echo " Fresh start.")"
echo "Check with: squeue -u $USER"
