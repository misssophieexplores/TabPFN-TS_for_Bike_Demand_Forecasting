#!/bin/bash
#SBATCH --job-name=tune-xgboost
#SBATCH --output=tune_xgboost_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00

source .venv/bin/activate
python -u forecasting/models/tuning/tune_xgboost.py --city london