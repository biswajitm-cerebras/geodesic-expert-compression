#!/bin/bash
#SBATCH --job-name=paper_reap_prune
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/paper_reap_prune_%A_%a.out
#SBATCH --error=logs/paper_reap_prune_%A_%a.out

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs
source "${REAP_DIR}/.venv/bin/activate"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(42 11 99)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
MODEL="Qwen/Qwen3-30B-A3B"
DATASET="theblackcat102/evol-codealpaca-v1"

printf 'Paper REAP reproduction\nmodel=%s\ndataset=%s\nseed=%s\ncompression=0.50\n' \
    "${MODEL}" "${DATASET}" "${SEED}"

python src/reap/prune.py \
    --model-name "${MODEL}" \
    --dataset-name "${DATASET}" \
    --compression-ratio 0.50 \
    --prune-method reap \
    --profile false \
    --do-eval false \
    --distance_measure cosine \
    --seed "${SEED}" \
    --output_file_name "observations_1024_cosine-seed_${SEED}.pt" \
    --batch_size 1 \
    --batches_per_category 1024 \
    --record_pruning_metrics_only true \
    --renormalize_router_weights true \
    --overwrite_observations false \
    --overwrite_pruned_model false \
    --smoke_test true

CHECKPOINT="${REAP_DIR}/artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_${SEED}-0.50"
test -f "${CHECKPOINT}/config.json"
find "${CHECKPOINT}" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .
echo "REAP checkpoint ready: ${CHECKPOINT}"
