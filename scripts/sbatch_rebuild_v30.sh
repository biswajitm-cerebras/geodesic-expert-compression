#!/bin/bash
#SBATCH --job-name=rebuild_v30
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/rebuild_v30_%j.out
#SBATCH --error=logs/rebuild_v30_%j.out

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source "${REAP_DIR}/.venv/bin/activate"

echo "=================================================================="
echo "Node:        $(hostname)"
echo "Rebuilding v30 with aliasing bug fix..."
echo "=================================================================="

rm -rf artifacts/adaptive_prune_merge_v30
timeout 10800 python scripts/adaptive_prune_merge.py \
    --keep-experts 64 \
    --output-dir artifacts/adaptive_prune_merge_v30

echo "=================================================================="
echo "Build complete. Files:"
ls -lh artifacts/adaptive_prune_merge_v30/
echo "=================================================================="
