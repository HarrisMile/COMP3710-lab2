#!/bin/bash
#SBATCH --job-name=oasis-gan
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=oasis_gan_%j.out
#SBATCH --error=oasis_gan_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part4_gan \
  --data-root /home/groups/comp3710/OASIS \
  --device cuda \
  --epochs 80 \
  --image-size 128 \
  --latent-dim 128 \
  --base-channels 64 \
  --batch-size 64 \
  --workers 4 \
  --no-amp \
  --output-dir results/part4_gan
