#!/bin/bash
#SBATCH --job-name=v27karcher_qwen3
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27karcher_qwen3_%j.out
#SBATCH --error=logs/v27karcher_qwen3_%j.out

# ---------------------------------------------------------------------------
# Iterative Karcher variant of the v27 Fisher-Geodesic expert merge on
# Qwen3-30B-A3B. Grouping (mc_smoe router-logit cosine) and the diagonal-Fisher
# metric are IDENTICAL to slurm_v27geodesic_qwen3.sbatch; the only change is the
# intra-cluster combination rule, which is a *superset* of v27 controlled by
# three env knobs read inside MoEExpertMerger._v27_fisher_geodesic_merge:
#
#   FISHER_BETA   metric temperature. 0 -> flat M-SMoE average, 1 -> v27,
#                 ->inf -> per-unit argmax (selection / pruning limit).
#   KARCHER_ITERS number of fixed-point iterations. 1 -> one-shot v27 mean.
#                 >1 -> Fisher-weighted Weiszfeld refinement toward the
#                 Fisher-Rao geodesic median (re-anchors the metric at the
#                 running midpoint; robust on geodesically WIDE clusters, which
#                 is where functional-subspace collapse is worst).
#   KARCHER_ETA   under-relaxation / damping (v27-GL lesson: MoE expert weights
#                 are all "body" MLP projections -> 0.5 tames the 2-cycle
#                 oscillation on curved segments).
#
# Defaults (ITERS=1, BETA=1, ETA=1) reproduce v27 bit-for-bit. This launcher
# sets the 10-step damped iteration.
#
# Merge-only run (observer -> cluster -> Fisher -> iterative merge -> save).
#
# Usage:
#   sbatch slurm_v27karcher_qwen3.sbatch [COMPRESSION_RATIO] [MODEL] [DATASET] [SEED] [FISHER_BATCHES] [OBS_BATCHES] [OBS_BATCH_SIZE] [OBS_MAXLEN]
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
# Observer workload -- kept IDENTICAL to the v27 baseline so grouping and the
# endpoint Fisher are directly comparable.
OBS_BATCHES="${6:-256}"
OBS_BATCH_SIZE="${7:-4}"
OBS_MAXLEN="${8:-1024}"

# --- iterative Karcher knobs (override at submit time via env if desired) -----
export KARCHER_ITERS="${KARCHER_ITERS:-10}"
export KARCHER_ETA="${KARCHER_ETA:-0.5}"
export FISHER_BETA="${FISHER_BETA:-1.0}"

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
echo "Merge method:    v27_fisher_geodesic (iterative Karcher)"
echo "KARCHER_ITERS:   ${KARCHER_ITERS}"
echo "KARCHER_ETA:     ${KARCHER_ETA}"
echo "FISHER_BETA:     ${FISHER_BETA}"
echo "=================================================================="

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
PORT=8000
echo "Num GPUs: ${NGPU}, vLLM port (unused in merge-only): ${PORT}"

CLUSTER_DESCRIPTION="v27_karcher"
OUTPUT_FILE_NAME="observations_${OBS_BATCHES}_cosine-seed_${SEED}_${CLUSTER_DESCRIPTION}.pt"
MERGED_MODEL_DIR_NAME="v27karcher-seed_${SEED}_${COMPRESSION_RATIO}"
SERVER_LOG="merging-v27karcher-seed_${SEED}.log"

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
echo "v27 iterative-Karcher merge finished. Merged model under:"
echo "  artifacts/${SHORT_MODEL}/${SHORT_DATASET}/non_uniform_merged_models/${MERGED_MODEL_DIR_NAME}/${CLUSTER_DESCRIPTION}"
echo "=================================================================="
