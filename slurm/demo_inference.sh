#!/bin/bash
#SBATCH --job-name=demo-infer
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=demo_inference_%j.out
#SBATCH --error=demo_inference_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate comp3710lab2
PROJECT_DIR="${PROJECT_DIR:-$HOME/COMP3710-Assignment2}"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
bash scripts/live_demo.sh summary
bash scripts/live_demo.sh lfw-infer
bash scripts/live_demo.sh vae-infer
bash scripts/live_demo.sh gan-infer
echo "Completed $(date)"
