#!/bin/bash
#SBATCH --job-name=v30_adaptive
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue

# Adaptive prune-vs-merge v30: merge high-redundancy layers, prune low-redundancy layers.
# All layers end with same K experts (servable, no config changes).

set -e
cd /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap

source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. /lustre/scratch/users/biswajit.mishra/.env

KEEP_EXPERTS=${KEEP_EXPERTS:-64}
REDUNDANCY_THRESHOLD=${REDUNDANCY_THRESHOLD:-0.15}
OBS_CACHE=${OBS_CACHE:-artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/observations_64_cosine-seed_42_v27_gl.pt}

mkdir -p artifacts/eval/v30_adaptive

echo "Running adaptive prune-vs-merge..."
python scripts/adaptive_prune_merge.py \
  --keep-experts $KEEP_EXPERTS \
  --redundancy-threshold $REDUNDANCY_THRESHOLD \
  --obs-cache "$OBS_CACHE" \
  --output-dir artifacts/adaptive_prune_merge_v30 \
  --output-name v30_adaptive 2>&1 | tee logs/adaptive_prune_merge.log

echo "✓ Adaptive merge complete. Compressed model: artifacts/adaptive_prune_merge_v30/model.safetensors"
echo "  Layer decisions: artifacts/adaptive_prune_merge_v30/adaptive_decisions.json"
