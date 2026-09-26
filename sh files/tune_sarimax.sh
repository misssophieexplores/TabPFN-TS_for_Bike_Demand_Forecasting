#!/bin/bash
#SBATCH --job-name=tune-sarimax
#SBATCH --output=tune_sarimax_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

source .venv/bin/activate
python -u forecasting/models/tuning/tune_sarimax.py --city london
python -u forecasting/models/tuning/tune_sarimax.py --city washington