#!/bin/bash
#SBATCH --job-name=tune-prophet
#SBATCH --output=tune_prophet_%j.log
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00

source .venv/bin/activate
python -u forecasting/models/tuning/tune_prophet.py --city london