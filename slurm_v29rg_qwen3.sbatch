#!/bin/bash
#SBATCH --job-name=v29rg_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=490G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v29rg_qwen3_%j.out
#SBATCH --error=logs/v29rg_qwen3_%j.out
# ---------------------------------------------------------------------------
# v29 = v27-GL Fisher-Geodesic merge + D3 ROUTER-GATED FISHER.
#
# Multiplicatively scales each expert's per-unit Fisher by its corpus router-gate
# mass g_i ** gamma BEFORE the sum-to-1 normalisation (read by merge.py via
# GL_ROUTER_GATED). This is the mean-field diagonal reduction of the REAP-style
# router-weighted functional loss:
#     argmin_theta sum_i sum_t g_{i,t} || f_theta(x_t) - f_theta_i(x_t) ||^2
#   gamma=0 -> exact v27 Fisher-Rao mean
#   gamma=1 -> Fisher weighted by how hard the router leans on each expert
#   gamma->inf -> per-unit collapse onto the highest-gate expert (REAP-prune limit)
#
# Reuses the v27_gl observer cache (identical M-SMoE grouping); only the merge
# blend changes. Node Fishers are recomputed (gating is applied after Fisher
# collection, so grouping/Fisher are unaffected).
#
# Usage:
#   sbatch slurm_v29rg_qwen3.sbatch [GAMMA] [COMPRESSION] [MODEL] [DATASET] \
#                                   [SEED] [FISHER_BATCHES] [GL_ITERS] [GL_NODES]
# ---------------------------------------------------------------------------

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

GAMMA="${1:-1.0}"
COMPRESSION_RATIO="${2:-0.5}"
MODEL="${3:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${4:-theblackcat102/evol-codealpaca-v1}"
SEED="${5:-42}"
FISHER_BATCHES="${6:-128}"
GL_ITERS_ARG="${7:-3}"
GL_NODES_ARG="${8:-3}"
# Observer workload (kept IDENTICAL to the v27-GL baseline so grouping is comparable).
OBS_BATCHES="${9:-64}"
OBS_BATCH_SIZE="${10:-4}"
OBS_MAXLEN="${11:-1024}"

# --- environment (compute nodes have no internet: use pre-populated HF cache) -
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1

# --- v27-GL path-integration config (read by main.py::_gl_config_from_env) ----
export GL_ITERS="${GL_ITERS_ARG}"
export GL_NODES="${GL_NODES_ARG}"
export GL_CONV_EPS="${GL_CONV_EPS:-1e-4}"
# --- D3 router-gated Fisher (read by merge.py::_v27_fisher_geodesic_merge) -----
export GL_ROUTER_GATED="${GAMMA}"

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
echo "GL_ROUTER_GATED: ${GL_ROUTER_GATED}  (0=v27 Fisher mean, >0 router-gated, inf=REAP prune)"
echo "Merge method:    v29 router-gated GL Fisher-Geodesic (M-SMoE grouping)"
echo "=================================================================="

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
PORT=8000
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

# Reuse the v27_gl observer cache (identical clustering) but write to a distinct
# merged-model dir so v29 does not collide with the v27-GL / v28 outputs.
CLUSTER_DESCRIPTION="v27_gl"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
GAMMA_TAG=$(echo "${GAMMA}" | tr '.' 'p')
MERGED_MODEL_DIR_NAME="v29rg${GAMMA_TAG}-seed_${SEED}_${COMPRESSION_RATIO}_it${GL_ITERS}_n${GL_NODES}"
SERVER_LOG="merging-v29rg${GAMMA_TAG}-seed_${SEED}.log"

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
echo "v29 router-gated merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
