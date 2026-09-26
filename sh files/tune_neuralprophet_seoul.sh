#!/bin/bash
#SBATCH --job-name=tune-np-seoul
#SBATCH --output=tune_neuralprophet_seoul_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

source .venv/bin/activate

python -u forecasting/models/tuning/tune_neuralprophet_fast.py \
    --city seoul \
    --trials 20 \
    --search-epochs 50 \
    --search-folds 2