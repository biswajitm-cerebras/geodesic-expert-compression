#!/bin/bash
#SBATCH --job-name=paper_collect
#SBATCH --account=cerebras
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/paper_collect_%j.out
#SBATCH --error=logs/paper_collect_%j.out

set -euo pipefail
ROOT="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${ROOT}"
source "${ROOT}/.venv/bin/activate"
python scripts/collect_paper_reproduction.py
