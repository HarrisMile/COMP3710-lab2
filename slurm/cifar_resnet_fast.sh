#!/bin/bash
#SBATCH --job-name=cifar-fast
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=cifar_fast_%j.out
#SBATCH --error=cifar_fast_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export TMPDIR="${SLURM_TMPDIR:-$HOME/tmp}"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
python -m comp3710_lab2.part3_cifar_resnet_fast \
  --device cuda \
  --epochs 72 \
  --batch-size 512 \
  --learning-rate 0.4 \
  --mixup-alpha 0.0 \
  --label-smoothing 0.1 \
  --cutout-size 8 \
  --ema-decay 0.99 \
  --ema-update-every 5 \
  --target-accuracy 0.94 \
  --stop-on-target \
  --compile \
  --output-dir results/part3_cifar_resnet_fast
