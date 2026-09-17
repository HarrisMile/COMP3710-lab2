#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python scripts/test_cuda.py
