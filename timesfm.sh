#!/bin/bash
#SBATCH --job-name=timesfm-poc
#SBATCH --output=TimesFM_POV/timesfm_poc_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

source .venv/bin/activate
python -u TimesFM_POV/timesfm_poc.py