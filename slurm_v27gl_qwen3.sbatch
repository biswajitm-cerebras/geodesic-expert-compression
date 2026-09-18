#!/bin/bash
#SBATCH --job-name=v27gl_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=490G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27gl_qwen3_%j.out
#SBATCH --error=logs/v27gl_qwen3_%j.out

# ---------------------------------------------------------------------------
# M-SMoE *grouping* + v27-GL *path-integrated* Fisher-Geodesic *merging*.
#
# Identical to slurm_v27geodesic_qwen3.sbatch (M-SMoE router-logits clustering,
# per-neuron Fisher blend) EXCEPT the Fisher is path-integrated along each
# expert's geodesic toward the running merged centroid using an n-point
# Gauss-Legendre quadrature, wrapped in a Picard fixed-point iteration:
#
#   for k in 1..GL_ITERS:                              # fixed-point iterations
#       for each expert i in a cluster:
#           gamma_i(t) = (1-t)*W_i + t*theta*_c        # geodesic node interpolant
#           F_hat_i(u) = sum_n w_n * F(gamma_i(t_n))   # GL quadrature of Fisher
#       theta*_c[u] = sum_i F_hat_i(u)*W_i[u] / sum_i F_hat_i(u)
#
# This is the direct MoE port of the model_merge `lws_GL_diag_v27` recipe
# (3-point GL nodes 0.5 +/- 0.5*sqrt(3/5), 0.5 with weights 5/18, 8/18, 5/18).
#
# COST: each GL iteration performs GL_NODES full-model backward passes over the
# Fisher calibration set, so total Fisher backprop cost ~= (1 + GL_ITERS*GL_NODES)x
# the endpoint merge. GL_ITERS=0 reduces to the plain endpoint v27 merge.
#
# Usage:
#   sbatch slurm_v27gl_qwen3.sbatch [COMPRESSION] [MODEL] [DATASET] [SEED] \
#                                   [FISHER_BATCHES] [GL_ITERS] [GL_NODES]
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
GL_ITERS_ARG="${6:-3}"
GL_NODES_ARG="${7:-3}"
# Observer workload (kept IDENTICAL to the baseline so grouping is comparable).
OBS_BATCHES="${8:-64}"
OBS_BATCH_SIZE="${9:-4}"
OBS_MAXLEN="${10:-1024}"

# --- environment (compute nodes have no internet: use pre-populated HF cache) -
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1

# --- v27-GL path-integration config (read by main.py::_gl_config_from_env) ----
export GL_ITERS="${GL_ITERS_ARG}"
export GL_NODES="${GL_NODES_ARG}"
export GL_CONV_EPS="${GL_CONV_EPS:-1e-4}"

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
echo "GL iters/nodes:  ${GL_ITERS} / ${GL_NODES}  (conv eps ${GL_CONV_EPS})"
echo "Merge method:    v27-GL path-integrated Fisher-Geodesic (M-SMoE grouping)"
echo "=================================================================="

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
PORT=8000
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

CLUSTER_DESCRIPTION="v27_gl"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
MERGED_MODEL_DIR_NAME="v27gl-seed_${SEED}_${COMPRESSION_RATIO}_it${GL_ITERS}_n${GL_NODES}"
SERVER_LOG="merging-v27gl-seed_${SEED}.log"

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
echo "v27-GL path-integrated merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
