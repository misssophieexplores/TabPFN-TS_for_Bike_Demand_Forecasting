#!/bin/bash
#SBATCH --job-name=bike-forecasting-v6_Prophets_Tuned
#SBATCH --output=bike-forecasting-v6_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --time=80:00:00

source .venv/bin/activate
python -u forecasting/main.py