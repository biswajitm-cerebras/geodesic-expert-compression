#!/bin/bash
# ---------------------------------------------------------------------------
# Submit REAP evals for the 3-way comparison:
#   1. dense parent (uncompressed reference)
#   2. M-SMoE baseline merge (frequency-weighted average)
#   3. v27 Fisher-Geodesic merge (per-neuron Fisher blend)
#
# Each eval waits (via --dependency=afterok) for its merge job if a job id is
# given. The dense parent has no dependency.
#
# Usage:
#   bash submit_evals.sh [MSMOE_MERGE_JID] [V27_MERGE_JID] [SEED]
# ---------------------------------------------------------------------------
set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"

MSMOE_JID="${1:-}"
V27_JID="${2:-}"
SEED="${3:-42}"

PARENT="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SHORT="Qwen3-30B-A3B-Instruct-2507"
DATASET_SHORT="evol-codealpaca-v1"
BASE="artifacts/${MODEL_SHORT}/${DATASET_SHORT}/non_uniform_merged_models"

MSMOE_DIR="${BASE}/m_smoe-seed_${SEED}_0.5/m_smoe"
V27_DIR="${BASE}/v27geo-seed_${SEED}_0.5/v27_geodesic"

submit() {
    local name="$1" model="$2" depjid="$3"
    local dep=()
    if [[ -n "${depjid}" ]]; then
        dep=(--dependency="afterok:${depjid}")
    fi
    echo ">>> submitting eval '${name}'  model=${model}  dep=${depjid:-none}"
    sbatch "${dep[@]}" --job-name="eval_${name}" \
        slurm_reap_eval.sbatch "${model}" "artifacts/eval/${name}" "${SEED}"
}

# 1) dense parent — no dependency, datasets/model already cached
submit "dense_parent"  "${PARENT}"   ""
# 2) M-SMoE baseline — after its merge job
submit "msmoe_0.5"     "${MSMOE_DIR}" "${MSMOE_JID}"
# 3) v27 Fisher-Geodesic — after its merge job
submit "v27geo_0.5"    "${V27_DIR}"   "${V27_JID}"

echo "--- queue ---"
squeue -u "$USER" -o "%.10i %.16j %.10T %.12E %R"
