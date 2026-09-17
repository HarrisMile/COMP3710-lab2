#!/bin/bash
#SBATCH --job-name=eigenfaces
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=eigenfaces_%j.out
#SBATCH --error=eigenfaces_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python -m comp3710_lab2.part2_eigenfaces \
  --data-home data \
  --components 150 \
  --trees 150 \
  --max-depth 15 \
  --max-features 150 \
  --output-dir results/part2_eigenfaces
