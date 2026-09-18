#!/bin/bash
#SBATCH --job-name=msmoe_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/msmoe_qwen3_%j.out
#SBATCH --error=logs/msmoe_qwen3_%j.out

# ---------------------------------------------------------------------------
# M-SMoE (Li et al., 2023) expert merging on Qwen3-30B-A3B, as implemented in
# the Cerebras REAP repo (arXiv 2510.13999). This runs ONLY the merge stage
# (observer -> cluster -> frequency-weighted merge -> save + smoke test).
# Evaluations are turned OFF here; run experiments/eval.sh separately.
#
# Usage:
#   sbatch slurm_msmoe_qwen3.sbatch [COMPRESSION_RATIO] [MODEL] [DATASET] [SEED]
# Defaults replicate the paper's 50% compression M-SMoE baseline.
# ---------------------------------------------------------------------------

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

COMPRESSION_RATIO="${1:-0.5}"
MODEL="${2:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${3:-theblackcat102/evol-codealpaca-v1}"
SEED="${4:-42}"
# Observer workload (the REAP default 1024x8x2048 runs ALL 128 experts on every
# token across 48 layers -> ~10 min/batch. These smaller values give plenty of
# router-logit statistics for clustering while finishing in a couple of hours.)
OBS_BATCHES="${5:-64}"
OBS_BATCH_SIZE="${6:-4}"
OBS_MAXLEN="${7:-1024}"

# --- environment (compute nodes have no internet: use pre-populated HF cache) -
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# datasets used by the observer are also cached ahead of time on the login node
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
echo "=================================================================="

# merging-cli.sh sets CUDA_VISIBLE_DEVICES from its first arg. Under SLURM the
# allocated GPUs are re-indexed to 0..N-1, so pass the local indices.
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
FIRST_DEVICE=0
PORT=$((8000 + FIRST_DEVICE))
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

# We call src/reap/main.py directly (the merge command from experiments/m_smoe.sh)
# instead of merging-cli.sh, because that chains experiments/eval.sh which starts
# a vLLM server unconditionally. Here we keep --do-eval false for a merge-only run.
CLUSTER_DESCRIPTION="m_smoe"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
MERGED_MODEL_DIR_NAME="m_smoe-seed_${SEED}_${COMPRESSION_RATIO}"
SERVER_LOG="merging-${FIRST_DEVICE}-seed_${SEED}.log"

python src/reap/main.py \
    --compression-ratio "${COMPRESSION_RATIO}" \
    --model-name "${MODEL}" \
    --dataset-name "${DATASET}" \
    --batches-per-category "${OBS_BATCHES}" \
    --batch-size "${OBS_BATCH_SIZE}" \
    --model-max-length "${OBS_MAXLEN}" \
    --merge-method frequency_weighted_average \
    --permute wm \
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
echo "M-SMoE merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
