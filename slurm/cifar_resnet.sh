#!/bin/bash
#SBATCH --job-name=cifar-resnet18
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=cifar_resnet_%j.out
#SBATCH --error=cifar_resnet_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part3_cifar_resnet \
  --device cuda \
  --epochs 100 \
  --batch-size 512 \
  --workers 4
