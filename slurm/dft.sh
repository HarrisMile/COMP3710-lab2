#!/bin/bash
#SBATCH --job-name=dft-gpu
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=dft_%j.out
#SBATCH --error=dft_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part1_dft \
  --device cuda \
  --sizes 64 128 256 512 1024 2048 \
  --repeats 3 \
  --output-dir results/part1_dft
