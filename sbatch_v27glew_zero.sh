#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Weight-wise (element-wise) v27_fisher_geodesic merge — SINGLE NODE
#
# Same recipe as sbatch_v27gl_paper_par.sh, but sets FISHER_ELEMENTWISE=1 so the
# blend uses the FULL [out, in] diagonal Fisher of every expert weight element
# instead of one scalar per output row. Each weight element is placed on its own
# Fisher-Rao geodesic between the cluster's experts, then Picard-iterated with a
# 3-point Gauss–Legendre quadrature (GL_ITERS=3).
#
# WHY SINGLE NODE:
#   The element-wise Fisher is a full [out, in] tensor per expert projection
#   (~58 GB fp16 for the Qwen3-30B expert set, vs ~0.03 GB for per-unit). Running
#   distributed would all_reduce a *fp32* copy on every rank → ~2× host-RAM blow-up
#   that OOMs a 515 GB node during the GL finalize. Single node keeps peak host RAM
#   at ~400 GB (originals fp32 + merged fp32 + gl_fisher fp16 + finalize transient).
#   Fisher is fp16-stored (downcast AFTER the fp32 L1-normalisation, so no overflow)
#   with a fp16-safe floor (F_FLOOR_EW=1e-4) so no expert weight underflows to zero.
#   Cost: ~2 h / Fisher pass × (1 seed + 3 nodes × 3 iters = 10) ≈ 20 h < 48 h.
#
# Calibration matches the paper run (identical clustering → reuses its obs file).
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=v27glew_zero
#SBATCH --partition=gpumid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/v27glew_zero_%j.out
#SBATCH --error=logs/v27glew_zero_%j.err

set -e
cd /lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap

source /lustre/scratch/users/biswajit.mishra/venv/bin/activate
. /lustre/scratch/users/biswajit.mishra/model_merge/.env
export PYTHONPATH="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap/src:${PYTHONPATH:-}"

mkdir -p logs

# ── Calibration (paper settings; identical → reuse the paper obs file) ─────────
BATCHES_PER_CAT=128          # 128 × batch_size=8 = 1024 samples (paper setting)
BATCH_SIZE=8
MODEL_MAX_LENGTH=2048
FISHER_NUM_BATCHES=1024
SEED=42
COMPRESSION_RATIO=0.5
OBS_FILE="observations_1024_cosine-seed_42_v27gl_paper.pt"

# ── Weight-wise variant env ────────────────────────────────────────────────────
export FISHER_ELEMENTWISE=1              # <-- full [out, in] Fisher per weight element
export ROUTER_COMPACT=zero               # <-- zero non-dominant router rows, keep 128 slots
export GL_ITERS=${GL_ITERS:-3}
export GL_NODES=${GL_NODES:-3}
export V28_FREQ_MIX=${V28_FREQ_MIX:-0.0}
export KARCHER_ITERS=${KARCHER_ITERS:-1} # MUST stay 1: >1 is undefined for element-wise Fisher
export KARCHER_ETA=${KARCHER_ETA:-1.0}

VARIANT_TAG="v27glew_zero"
MERGED_MODEL_DIR="${VARIANT_TAG}-paper-seed_${SEED}_${COMPRESSION_RATIO}_it${GL_ITERS}_n${GL_NODES}"

echo "========================================================"
echo "  Variant:          ${VARIANT_TAG}  (FISHER_ELEMENTWISE=1, ROUTER_COMPACT=zero)"
echo "  Merged model dir: ${MERGED_MODEL_DIR}"
echo "  Obs file:         ${OBS_FILE}"
echo "  GL_ITERS=${GL_ITERS}  GL_NODES=${GL_NODES}  KARCHER_ITERS=${KARCHER_ITERS}"
echo "  Node: $(hostname)   free RAM: $(free -g | awk '/Mem:/{print $7\" GB\"}')"
echo "========================================================"

COMMON_ARGS=(
    --model_name          "Qwen/Qwen3-30B-A3B-Instruct-2507"
    --dataset_name        "theblackcat102/evol-codealpaca-v1"
    --batches_per_category ${BATCHES_PER_CAT}
    --batch_size          ${BATCH_SIZE}
    --model_max_length    ${MODEL_MAX_LENGTH}
    --output_file_name    "${OBS_FILE}"
    --seed                ${SEED}
)

# ── PHASE 1: observer (only if the obs file is missing — reuse the paper run) ──
OBS_PATH="artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/all/${OBS_FILE}"
if [[ -f "${OBS_PATH}" ]]; then
    echo "── Phase 1 skipped: reusing existing observations ${OBS_PATH}"
    ls -lh "${OBS_PATH}"
else
    echo "── Phase 1: single-node observer (obs file missing) ──────────────────"
    python observer_par.py "${COMMON_ARGS[@]}" \
        2>&1 | tee "logs/v27glew_obs_${SLURM_JOB_ID}.log"
fi

# ── PHASE 2: Clustering + element-wise Fisher + Merge (single node, 4 GPUs) ────
# SLURM_NTASKS=1 → main.py runs single-process (no Gloo group, no all_reduce).
echo ""
echo "── Phase 2: Clustering + element-wise Fisher + Merge (1 node × 4 GPUs) ─"

python src/reap/main.py \
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
    --fisher_overwrite    true \
    --smoke_test          false \
    --do_eval             false \
    --overwrite_merged_model true \
    --plot_clusters       true \
    2>&1 | tee "logs/v27glew_zero_merge_${SLURM_JOB_ID}.log"

echo "✓ Done: artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/non_uniform_merged_models/${MERGED_MODEL_DIR}"
