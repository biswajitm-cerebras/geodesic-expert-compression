#!/bin/bash
#SBATCH --job-name=v27geo_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27geo_qwen3_%j.out
#SBATCH --error=logs/v27geo_qwen3_%j.out

# ---------------------------------------------------------------------------
# M-SMoE *grouping* + v27 Fisher-Geodesic *merging* on Qwen3-30B-A3B.
#
# Grouping is IDENTICAL to the M-SMoE baseline (router-logits cosine
# clustering via mc_smoe, dominant expert per cluster). Only the intra-cluster
# combination rule is replaced: instead of a frequency-weighted average, the
# experts of each cluster are blended with the per-neuron diagonal-Fisher rule
# from the model_merge v27 recipe:
#     W_merged[u] = sum_i F_i[u] * W_i[u] / sum_i F_i[u]
# where F_i[u] is the L1-normalised diagonal Fisher of expert i's output row u.
#
# Weight permutation is disabled for this method (the Fisher blend performs the
# neuron alignment implicitly, matching v27 which never permutes).
#
# Merge-only run (observer -> cluster -> Fisher -> merge -> save + smoke test).
#
# Usage:
#   sbatch slurm_v27geodesic_qwen3.sbatch [COMPRESSION_RATIO] [MODEL] [DATASET] [SEED] [FISHER_BATCHES]
# ---------------------------------------------------------------------------

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

COMPRESSION_RATIO="${1:-0.5}"
MODEL="${2:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${3:-theblackcat102/evol-codealpaca-v1}"
SEED="${4:-42}"
FISHER_BATCHES="${5:-128}"
# Observer workload (see slurm_msmoe_qwen3.sbatch note). Kept IDENTICAL to the
# baseline so grouping is directly comparable.
OBS_BATCHES="${6:-64}"
OBS_BATCH_SIZE="${7:-4}"
OBS_MAXLEN="${8:-1024}"

# --- environment (compute nodes have no internet: use pre-populated HF cache) -
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1

source "${REAP_DIR}/.venv/bin/activate"
set -a; source "${REAP_DIR}/.env"; set +a

echo "=================================================================="
echo "Node:            $(hostname)"
echo "GPUs visible:    ${CUDA_VISIBLE_DEVICES:-<from SLURM>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "Model:           ${MODEL}"
echo "Dataset:         ${DATASET}"
echo "Compression:     ${COMPRESSION_RATIO}"
echo "Seed:            ${SEED}"
echo "Fisher batches:  ${FISHER_BATCHES}"
echo "Merge method:    v27_fisher_geodesic (M-SMoE grouping)"
echo "=================================================================="

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
PORT=8000
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

CLUSTER_DESCRIPTION="v27_geodesic"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
MERGED_MODEL_DIR_NAME="v27geo-seed_${SEED}_${COMPRESSION_RATIO}"
SERVER_LOG="merging-v27geo-seed_${SEED}.log"

python src/reap/main.py \
    --compression-ratio "${COMPRESSION_RATIO}" \
    --model-name "${MODEL}" \
    --dataset-name "${DATASET}" \
    --batches-per-category "${OBS_BATCHES}" \
    --batch-size "${OBS_BATCH_SIZE}" \
    --model-max-length "${OBS_MAXLEN}" \
    --merge-method v27_fisher_geodesic \
    --fisher_num_batches "${FISHER_BATCHES}" \
    --cluster-method mc_smoe \
    --profile false \
    --server_log_file_name "${SERVER_LOG}" \
    --vllm-port "${PORT}" \
    --expert-sim router_logits \
    --distance_measure cosine \
    --frequency-penalty false \
    --merged-model-dir-name "${MERGED_MODEL_DIR_NAME}" \
    --cluster-description "${CLUSTER_DESCRIPTION}" \
    --do-eval false \
    --seed "${SEED}" \
    --output-file-name "${OUTPUT_FILE_NAME}" \
    --overwrite-merged-model true \
    --singleton_super_experts false \
    --singleton_outlier_experts false

SHORT_MODEL=$(echo "${MODEL}" | cut -d'/' -f2)
SHORT_DATASET=$(echo "${DATASET}" | cut -d'/' -f2)
echo "=================================================================="
echo "v27 Fisher-Geodesic merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
