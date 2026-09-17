#!/bin/bash
#SBATCH --job-name=oasis-v256
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=oasis_vae_256_%j.out
#SBATCH --error=oasis_vae_256_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part4_vae \
  --data-root /home/groups/comp3710/OASIS \
  --device cuda \
  --image-size 256 \
  --latent-dim 2 \
  --epochs 40 \
  --batch-size 32 \
  --workers 4 \
  --beta 1.0 \
  --output-dir results/part4_vae_beta1_256
