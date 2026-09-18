#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Paper-matching v27_fisher_geodesic merge — PARALLEL OBSERVER + PARALLEL FISHER
#
# Phase 1 (OBSERVER_NODES=2 nodes × 4 GPUs = 8 GPUs):
#   srun observer_par.py  →  each node processes 64 of 128 batches in parallel
#   → cuts observer wall-time from ~21 h to ~11 h.
#
# Phase 2 (SLURM_NNODES nodes × 4 GPUs, default 4 nodes = 16 GPUs):
#   srun main.py on all allocated nodes — obs file already exists, observer
#   skipped; each node runs one copy of the model (device_map=auto, 4 GPUs) and
#   processes 1/N of the Fisher calibration batches. Gloo all_reduce sums the
#   squared-gradient accumulators, giving exact N× Fisher speedup.
#   4 nodes → ~30 min/Fisher node (vs ~2 h single-node) → ~3–4 h total.
#
# Total (4 nodes): ~5 h  (vs ~13 h phase-2-only single-node).
#
# Submit with more nodes for faster Fisher (up to available idle nodes):
#   sbatch --nodes=8 sbatch_v27gl_paper_par.sh   # ~2 h Fisher per node
#
# Calibration matches paper (arXiv:2510.13999, Table A6/A7, ≤110B):
#   obs:    128 batches × bs=8 × seq=2048  →  1024 samples × 2048 tokens
#   Fisher: 1024 batches × bs=1            →  1024 samples × 2048 tokens
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=v27gl_paper_par
#SBATCH --partition=gpumid
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27gl_paper_par_%j.out
#SBATCH --error=logs/v27gl_paper_par_%j.err

set -e
cd /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap

source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. /lustre/scratch/users/biswajit.mishra/model_merge/.env
export PYTHONPATH="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap/src:${PYTHONPATH:-}"

mkdir -p logs

# ── Calibration (paper settings) ─────────────────────────────────────────────
BATCHES_PER_CAT=128          # 128 × batch_size=8 = 1024 samples (paper setting)
BATCH_SIZE=8                 # bs=8 is the max that fits (model=47GB, batch~16GB, total~63GB < 79GB)
MODEL_MAX_LENGTH=2048
FISHER_NUM_BATCHES=1024      # Fisher batch_size=1 hardcoded → 1024 samples
SEED=42
COMPRESSION_RATIO=0.5
OBS_FILE="observations_1024_cosine-seed_42_v27gl_paper.pt"

# ── Variant env vars (override at submit time if needed) ──────────────────────
export GL_ITERS=${GL_ITERS:-3}
export GL_NODES=${GL_NODES:-3}
export V28_FREQ_MIX=${V28_FREQ_MIX:-0.0}
export KARCHER_ITERS=${KARCHER_ITERS:-1}
export KARCHER_ETA=${KARCHER_ETA:-1.0}

# ── Output name ───────────────────────────────────────────────────────────────
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
echo "  Nodes: ${SLURM_NNODES}  Tasks: ${SLURM_NTASKS}"
echo "========================================================"

# ── Common python args ────────────────────────────────────────────────────────
COMMON_ARGS=(
    --model_name          "Qwen/Qwen3-30B-A3B-Instruct-2507"
    --dataset_name        "theblackcat102/evol-codealpaca-v1"
    --batches_per_category ${BATCHES_PER_CAT}
    --batch_size          ${BATCH_SIZE}
    --model_max_length    ${MODEL_MAX_LENGTH}
    --output_file_name    "${OBS_FILE}"
    --seed                ${SEED}
)

# ── PHASE 1: Distributed observer (always 2 nodes) ─────────────────────────────
OBSERVER_NODES=2
echo ""
echo "── Phase 1: Distributed observer (${OBSERVER_NODES} nodes × 4 GPUs) ──────"
echo "   Each node processes $((BATCHES_PER_CAT / OBSERVER_NODES)) batches in parallel."

MASTER_HOST=$(scontrol show hostname "${SLURM_NODELIST}" | head -1)
# The short gpumid hostname resolves to link-local IPv6 first. Use its
# routable InfiniBand IPv4 address explicitly for the distributed rendezvous.
MASTER_ADDR=$(getent ahostsv4 "${MASTER_HOST}" | awk '$2 == "STREAM" {print $1; exit}')
if [[ -z "${MASTER_ADDR}" ]]; then
    echo "ERROR: could not resolve ${MASTER_HOST} to an IPv4 address" >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=ibs9

echo "   Rendezvous: ${MASTER_HOST} (${MASTER_ADDR}:${MASTER_PORT}) via ${GLOO_SOCKET_IFNAME}"

srun --ntasks=${OBSERVER_NODES} --nodes=${OBSERVER_NODES} python observer_par.py \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "logs/v27gl_paper_par_obs_${SLURM_JOB_ID}.log"

echo "── Phase 1 complete. Obs file should exist now. ─────────────────────────"
ls -lh "artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/${OBS_FILE}"

# ── PHASE 2: Clustering + Fisher + Merge (all allocated nodes, distributed) ──
# main.py detects SLURM_NTASKS > 1 and inits a Gloo process group automatically.
# Each node loads a full copy of the model (device_map=auto, 4 GPUs) and
# processes 1/SLURM_NNODES of the Fisher calibration batches.
# Non-rank-0 nodes skip all disk writes; rank-0 saves the merged model.
echo ""
echo "── Phase 2: Clustering + Fisher + Merge (${SLURM_NNODES} nodes × 4 GPUs) ─"
echo "   Each node processes $((FISHER_NUM_BATCHES / SLURM_NNODES)) Fisher batches in parallel."

export MASTER_PORT=29600    # different port from phase 1 to avoid TIMED_WAIT

srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 python src/reap/main.py \
    "${COMMON_ARGS[@]}" \
    --compression_ratio   ${COMPRESSION_RATIO} \
    --cluster_method      mc_smoe \
    --expert_sim          router_logits \
    --distance_measure    cosine \
    --linkage_method      average \
    --cluster_description "${VARIANT_TAG}_paper" \
    --merge_method        v27_fisher_geodesic \
    --merged_model_dir_name "${MERGED_MODEL_DIR}" \
    --fisher_num_batches  ${FISHER_NUM_BATCHES} \
    --smoke_test          false \
    --do_eval             false \
    --overwrite_merged_model true \
    --plot_clusters       true \
    2>&1 | tee "logs/v27gl_paper_par_merge_${SLURM_JOB_ID}.log"

echo "✓ Done: artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/non_uniform_merged_models/${MERGED_MODEL_DIR}"
