#!/bin/bash
#SBATCH --job-name=tune-arima
#SBATCH --output=tune_arima_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00

source .venv/bin/activate
python -u forecasting/models/tuning/tune_arima.py --city london
python -u forecasting/models/tuning/tune_arima.py --city washington