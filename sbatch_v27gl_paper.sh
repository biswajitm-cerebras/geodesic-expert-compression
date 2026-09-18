#!/bin/bash
#SBATCH --job-name=v27gl_paper
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27gl_paper_%j.out
#SBATCH --error=logs/v27gl_paper_%j.err

# v27_fisher_geodesic merge with paper-matching calibration:
#   obs:    32 batches × batch_size=32 × seq_len=2048  →  1024 samples × 2048 tokens
#   Fisher: 1024 batches × batch_size=1  × seq_len=2048 →  1024 samples × 2048 tokens
# batch_size=32 (vs paper's 8) reduces observer forward passes 4×; same total 1024 samples.
# gpuhigh (8 GPUs) vs gpumid (4 GPUs) halves per-pass time → ~8× faster observer overall.
# Matches Table A6/A7 of arXiv:2510.13999 (REAP, ICLR 2026) for ≤50B models.
#
# Runs v27gl (pure Fisher geodesic, no freq_mix, no Karcher):
#   GL_ITERS=3, GL_NODES=3 (same as original smoke-test run)
#
# To run v28fm0p5 (freq_mix=0.5) on top of the same obs_cache, use:
#   V28_FREQ_MIX=0.5 sbatch --export=ALL sbatch_v27gl_paper.sh
# To run v29rg2p0 (2 Karcher iters):
#   KARCHER_ITERS=2 sbatch --export=ALL sbatch_v27gl_paper.sh

set -e
cd /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap

source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. /lustre/scratch/users/biswajit.mishra/model_merge/.env
export PYTHONPATH="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap/src:${PYTHONPATH:-}"

mkdir -p logs

# ── Calibration (paper settings) ─────────────────────────────────────────────
BATCHES_PER_CAT=128          # 128 × batch_size=8 = 1024 samples (paper setting)
BATCH_SIZE=8                 # bs=16/32 OOM (model=47GB + batch>31GB > 79GB); bs=8 safe (~16GB activation)
MODEL_MAX_LENGTH=2048
FISHER_NUM_BATCHES=1024      # Fisher batch_size=1 hardcoded → 1024 samples
SEED=42
COMPRESSION_RATIO=0.5
OBS_FILE="observations_1024_cosine-seed_42_v27gl_paper.pt"

# ── Variant env vars (override at submit time if needed) ──────────────────────
# v27gl default: GL_ITERS=3, GL_NODES=3, V28_FREQ_MIX=0, KARCHER_ITERS=1
export GL_ITERS=${GL_ITERS:-3}
export GL_NODES=${GL_NODES:-3}
export V28_FREQ_MIX=${V28_FREQ_MIX:-0.0}
export KARCHER_ITERS=${KARCHER_ITERS:-1}
export KARCHER_ETA=${KARCHER_ETA:-1.0}

# ── Output name (auto-label from variant env vars) ───────────────────────────
VARIANT_TAG="v27gl"
if [[ "${V28_FREQ_MIX}" != "0.0" && "${V28_FREQ_MIX}" != "0" ]]; then
    FM=$(echo "${V28_FREQ_MIX}" | sed 's/\.//g')
    VARIANT_TAG="v28fm${FM}"
fi
if [[ "${KARCHER_ITERS}" != "1" ]]; then
    RG=$(echo "${KARCHER_ITERS}" | sed 's/\.//g')
    VARIANT_TAG="v29rg${RG}p0"
fi
MERGED_MODEL_DIR="${VARIANT_TAG}-paper-seed_${SEED}_${COMPRESSION_RATIO}_it${GL_ITERS}_n${GL_NODES}"

echo "========================================================"
echo "  Variant:          ${VARIANT_TAG}"
echo "  Merged model dir: ${MERGED_MODEL_DIR}"
echo "  Obs file:         ${OBS_FILE}"
echo "  GL_ITERS=${GL_ITERS}  GL_NODES=${GL_NODES}"
echo "  V28_FREQ_MIX=${V28_FREQ_MIX}  KARCHER_ITERS=${KARCHER_ITERS}"
echo "  obs: ${BATCHES_PER_CAT} batches × bs=${BATCH_SIZE} × len=${MODEL_MAX_LENGTH}"
echo "  Fisher: ${FISHER_NUM_BATCHES} samples × len=${MODEL_MAX_LENGTH}"
echo "========================================================"

python src/reap/main.py \
    --model_name          "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --dataset_name        "theblackcat102/evol-codealpaca-v1" \
    --compression_ratio   ${COMPRESSION_RATIO} \
    --cluster_method      mc_smoe \
    --expert_sim          router_logits \
    --distance_measure    cosine \
    --linkage_method      average \
    --cluster_description "${VARIANT_TAG}_paper" \
    --merge_method        v27_fisher_geodesic \
    --merged_model_dir_name "${MERGED_MODEL_DIR}" \
    --fisher_num_batches  ${FISHER_NUM_BATCHES} \
    --batches_per_category ${BATCHES_PER_CAT} \
    --batch_size          ${BATCH_SIZE} \
    --model_max_length    ${MODEL_MAX_LENGTH} \
    --output_file_name    "${OBS_FILE}" \
    --seed                ${SEED} \
    --smoke_test          false \
    --do_eval             false \
    --overwrite_merged_model true \
    --plot_clusters       true \
    2>&1 | tee "logs/v27gl_paper_${SLURM_JOB_ID}.log"

echo "✓ Merge complete: artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/non_uniform_merged_models/${MERGED_MODEL_DIR}"
