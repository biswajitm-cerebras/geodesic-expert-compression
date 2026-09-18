#!/bin/bash
#SBATCH --job-name=fg_cluster
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/fisher_geodesic_cluster_%j.out
#SBATCH --error=logs/fisher_geodesic_cluster_%j.err

# Router-weighted local Fisher--Rao expert clustering followed by a
# path-integrated Fisher barycenter. Pairwise clustering uses the practical
# endpoint-average local metric; GL quadrature is used for each cluster center.
#
# Optional submit-time overrides, for example:
#   GL_ITERS=0 FISHER_SKETCH_SIZE=0 sbatch --export=ALL sbatch_fisher_geodesic_cluster.sh

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. /lustre/scratch/users/biswajit.mishra/model_merge/.env
export PYTHONPATH="${REAP_DIR}/src:${PYTHONPATH:-}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
DATASET="${DATASET:-theblackcat102/evol-codealpaca-v1}"
SEED="${SEED:-42}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
BATCHES_PER_CATEGORY="${BATCHES_PER_CATEGORY:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
FISHER_NUM_BATCHES="${FISHER_NUM_BATCHES:-1024}"
FISHER_SKETCH_SIZE="${FISHER_SKETCH_SIZE:-32}"
FISHER_WEIGHT="${FISHER_WEIGHT:-1.0}"
ROUTER_WEIGHT="${ROUTER_WEIGHT:-0.25}"
ACTIVATION_WEIGHT="${ACTIVATION_WEIGHT:-0.25}"
FISHER_IMPORTANCE_SOURCE="${FISHER_IMPORTANCE_SOURCE:-reap}"
OBS_FILE="${OBS_FILE:-observations_1024_cosine-seed_${SEED}_fisher_cluster.pt}"

MODEL_SLUG="${MODEL##*/}"
DATASET_SLUG="${DATASET##*/}"
CACHE_DIR="${REAP_DIR}/artifacts/fisher_geodesic_cache/${MODEL_SLUG}/${DATASET_SLUG}"
mkdir -p "${CACHE_DIR}"
FISHER_CACHE_PATH="${FISHER_CACHE_PATH:-${CACHE_DIR}/endpoint_fisher_b${FISHER_NUM_BATCHES}_seed${SEED}.pt}"
FISHER_DISTANCE_CACHE_PATH="${FISHER_DISTANCE_CACHE_PATH:-${CACHE_DIR}/distance_b${FISHER_NUM_BATCHES}_s${FISHER_SKETCH_SIZE}_seed${SEED}.pt}"

export GL_ITERS="${GL_ITERS:-3}"
export GL_NODES="${GL_NODES:-3}"
export GL_CONV_EPS="${GL_CONV_EPS:-1e-4}"
export FISHER_ELEMENTWISE="${FISHER_ELEMENTWISE:-0}"
export FISHER_BETA="${FISHER_BETA:-1.0}"
export KARCHER_ITERS="${KARCHER_ITERS:-1}"
export KARCHER_ETA="${KARCHER_ETA:-1.0}"
export V28_FREQ_MIX="${V28_FREQ_MIX:-0.0}"
# Explicit REAP conditioning is supplied through --fisher_router_weight_source;
# leave the older frequency-only gate disabled to avoid double weighting.
export GL_ROUTER_GATED="${GL_ROUTER_GATED:-0.0}"
export ROUTER_COMPACT="${ROUTER_COMPACT:-slice}"

VARIANT_TAG="fgcluster-reap-localfr-gl"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-${VARIANT_TAG}-seed_${SEED}-${COMPRESSION_RATIO}}"

echo "========================================================"
echo "  Model:              ${MODEL}"
echo "  Dataset:            ${DATASET}"
echo "  Compression:        ${COMPRESSION_RATIO}"
echo "  Fisher batches:     ${FISHER_NUM_BATCHES}"
echo "  Coordinate sketch:  ${FISHER_SKETCH_SIZE}"
echo "  Component weights:  F=${FISHER_WEIGHT} R=${ROUTER_WEIGHT} A=${ACTIVATION_WEIGHT}"
echo "  Fisher importance:  ${FISHER_IMPORTANCE_SOURCE}"
echo "  GL:                  iterations=${GL_ITERS}, nodes=${GL_NODES}"
echo "  Router compaction:   ${ROUTER_COMPACT}"
echo "  Fisher cache:        ${FISHER_CACHE_PATH}"
echo "  Distance cache:      ${FISHER_DISTANCE_CACHE_PATH}"
echo "========================================================"

python src/reap/main.py \
    --model_name "${MODEL}" \
    --dataset_name "${DATASET}" \
    --compression_ratio "${COMPRESSION_RATIO}" \
    --cluster_method agglomerative \
    --expert_sim fisher_geodesic \
    --linkage_method average \
    --frequency_penalty false \
    --fisher_distance_weight "${FISHER_WEIGHT}" \
    --router_distance_weight "${ROUTER_WEIGHT}" \
    --activation_distance_weight "${ACTIVATION_WEIGHT}" \
    --fisher_sketch_size "${FISHER_SKETCH_SIZE}" \
    --cluster_description "${VARIANT_TAG}" \
    --merge_method v27_fisher_geodesic \
    --fisher_router_weight_source "${FISHER_IMPORTANCE_SOURCE}" \
    --fisher_num_batches "${FISHER_NUM_BATCHES}" \
    --fisher_cache_path "${FISHER_CACHE_PATH}" \
    --fisher_distance_cache_path "${FISHER_DISTANCE_CACHE_PATH}" \
    --merged_model_dir_name "${MERGED_MODEL_DIR}" \
    --batches_per_category "${BATCHES_PER_CATEGORY}" \
    --batch_size "${BATCH_SIZE}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --output_file_name "${OBS_FILE}" \
    --seed "${SEED}" \
    --smoke_test false \
    --do_eval false \
    --overwrite_merged_model true \
    --plot_clusters true \
    2>&1 | tee "logs/fisher_geodesic_cluster_${SLURM_JOB_ID}.log"

echo "Fisher-geodesic expert clustering and merge complete."