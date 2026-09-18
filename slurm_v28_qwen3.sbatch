#!/bin/bash
#SBATCH --job-name=v28_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=490G
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v28_qwen3_%j.out
#SBATCH --error=logs/v28_qwen3_%j.out

# ---------------------------------------------------------------------------
# v28 = v27-GL path-integrated Fisher-Geodesic merge + DOMINANT-PRESERVING damping.
#
# Identical pipeline to slurm_v27gl_qwen3.sbatch (M-SMoE router-logits grouping,
# per-neuron path-integrated Fisher blend, Picard fixed-point) EXCEPT the final
# per-unit Fisher blend weights are convex-interpolated toward the M-SMoE
# usage-frequency prior via the V28_FREQ_MIX knob (read in merge.py::
# _v27_fisher_geodesic_merge):
#
#     Fw' = (1 - mix) * Fw + mix * p ,   p_i = proba_i / sum_j proba_j
#
#   mix=0.0 -> pure v27 Fisher-Rao mean (identical to slurm_v27gl_qwen3.sbatch)
#   mix=1.0 -> M-SMoE frequency-weighted mean (dominant expert preserved)
#   0<mix<1 -> interpolate; damps the Fisher-averaging that dilutes the dominant
#              expert on geodesically wide MoE clusters (the v27geo failure mode).
#
# REUSES the v27_gl observer cache (same CLUSTER_DESCRIPTION -> identical grouping);
# only the merge blend changes. Node Fishers are recomputed (freq_mix is applied
# after Fisher collection, so grouping/Fisher are unaffected).
#
# Usage:
#   sbatch slurm_v28_qwen3.sbatch [FREQ_MIX] [COMPRESSION] [MODEL] [DATASET] \
#                                 [SEED] [FISHER_BATCHES] [GL_ITERS] [GL_NODES]
# ---------------------------------------------------------------------------

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

FREQ_MIX="${1:-0.5}"
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
# --- v28 dominant-preserving damping (read by merge.py) -----------------------
export V28_FREQ_MIX="${FREQ_MIX}"

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
echo "V28_FREQ_MIX:    ${V28_FREQ_MIX}  (0=v27 Fisher mean, 1=M-SMoE freq mean)"
echo "Merge method:    v28 dominant-preserving GL Fisher-Geodesic (M-SMoE grouping)"
echo "=================================================================="

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
PORT=8000
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

# Reuse the v27_gl observer cache (identical clustering) but write to a distinct
# merged-model dir so v28 does not collide with the v27-GL output.
CLUSTER_DESCRIPTION="v27_gl"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
MIX_TAG=$(echo "${FREQ_MIX}" | tr '.' 'p')
MERGED_MODEL_DIR_NAME="v28fm${MIX_TAG}-seed_${SEED}_${COMPRESSION_RATIO}_it${GL_ITERS}_n${GL_NODES}"
SERVER_LOG="merging-v28fm${MIX_TAG}-seed_${SEED}.log"

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
echo "v28 dominant-preserving merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
