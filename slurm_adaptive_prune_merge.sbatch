#!/bin/bash
#SBATCH --job-name=v30_adaptive_merge
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -e
cd /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap
source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. .env

KEEP_EXPERTS=${1:-64}
REDUNDANCY_THRESHOLD=${2:-0.15}

mkdir -p logs artifacts/adaptive_prune_merge_v30

echo "======================================================================"
echo " V30: ADAPTIVE PRUNE-VS-MERGE (Qwen3-30B compression to $KEEP_EXPERTS experts)"
echo "======================================================================"
echo "Strategy:"
echo "  - Layer 0 & 47: HIGH redundancy (act-cos 0.27, 0.39) → MERGE clusters"
echo "  - Layers 1-46: LOW redundancy (orthogonal, <0.15) → PRUNE by saliency"
echo "  - Result: Uniform $KEEP_EXPERTS experts per layer (vLLM-compatible)"
echo ""
echo "Configuration:"
echo "  Keep experts: $KEEP_EXPERTS per layer"
echo "  Redundancy threshold: $REDUNDANCY_THRESHOLD"
echo "  Observer cache: v27_gl (64 batch size)"
echo "  Device map: auto (4 GPUs)"
echo ""

time python scripts/adaptive_prune_merge.py \
  --keep-experts "$KEEP_EXPERTS" \
  --redundancy-threshold "$REDUNDANCY_THRESHOLD" \
  --obs-cache artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/observations_64_cosine-seed_42_v27_gl.pt \
  --output-dir artifacts/adaptive_prune_merge_v30 2>&1 | tee logs/adaptive_prune_merge.log

echo ""
echo "======================================================================"
echo " ✓ MERGE COMPLETE"
echo "======================================================================"
echo "  Checkpoint: artifacts/adaptive_prune_merge_v30/"
echo "  Config: artifacts/adaptive_prune_merge_v30/config.json (num_experts=$KEEP_EXPERTS)"
echo "  Weights: artifacts/adaptive_prune_merge_v30/model.safetensors"
echo "  Decisions: artifacts/adaptive_prune_merge_v30/adaptive_decisions.json"
echo ""
echo "Next: Run evaluation with:"
echo "  sbatch slurm_reap_eval.sbatch artifacts/adaptive_prune_merge_v30"
echo ""

