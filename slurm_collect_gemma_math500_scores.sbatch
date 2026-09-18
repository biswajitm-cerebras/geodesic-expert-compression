#!/usr/bin/env bash
#SBATCH --job-name=collect_gemma_math
#SBATCH --partition=standard
#SBATCH --account=cerebras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/scratch/users/biswajit.mishra/model_merge/logs/collect_gemma_math_%j.out
#SBATCH --error=/lustre/scratch/users/biswajit.mishra/model_merge/logs/collect_gemma_math_%j.err

set -euo pipefail

python /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap/scripts/collect_gemma_math500_scores.py