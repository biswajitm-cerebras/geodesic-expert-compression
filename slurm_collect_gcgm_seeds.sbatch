#!/bin/bash
#SBATCH --job-name=gcgm_collect
#SBATCH --account=cerebras
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=logs/gcgm_collect_%j.out
#SBATCH --error=logs/gcgm_collect_%j.out

set -euo pipefail

ROOT="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${ROOT}"
source "${ROOT}/.venv/bin/activate"
python scripts/collect_gcgm_seeds.py