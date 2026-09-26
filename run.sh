#!/bin/bash
#SBATCH --job-name=bike-forecasting-v5
#SBATCH --output=bike-forecasting-v5_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00

source .venv/bin/activate
python -u forecasting/main.py