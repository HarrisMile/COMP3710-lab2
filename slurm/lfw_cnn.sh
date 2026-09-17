#!/bin/bash
#SBATCH --job-name=lfw-cnn
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=lfw_cnn_%j.out
#SBATCH --error=lfw_cnn_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part3_lfw_cnn \
  --data-home data \
  --device cuda \
  --epochs 50 \
  --batch-size 64 \
  --workers 2 \
  --rf-metrics results/part2_eigenfaces/metrics.json \
  --output-dir results/part3_lfw_cnn
