#!/bin/bash
#SBATCH --job-name=oasis-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=oasis_unet_%j.out
#SBATCH --error=oasis_unet_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part4_unet \
  --data-root /home/groups/comp3710/OASIS \
  --device cuda \
  --epochs 50 \
  --batch-size 8 \
  --workers 4
