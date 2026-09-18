#!/bin/bash
#SBATCH --job-name=reap_eval
#SBATCH --account=cerebras
#SBATCH --partition=gpumid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/reap_eval_%j.out
#SBATCH --error=logs/reap_eval_%j.out

# ---------------------------------------------------------------------------
# Evaluate a (merged or dense) Qwen3-MoE model with the REAP harness:
#   - lm-eval QA:  winogrande, arc_challenge, arc_easy, boolq, hellaswag,
#                  mmlu, openbookqa, rte      (0-shot)
#   - EvalPlus:    humaneval(+),  mbpp(+)
#   - evalscope math: gsm8k, math_500
# LiveCodeBench and WildBench are disabled (heavy / fragile offline deps).
#
# Serves the model with vLLM (TP=4) and evaluates via the local server.
#
# Usage:
#   sbatch slurm_reap_eval.sbatch <MODEL_DIR_OR_HF_ID> [RESULTS_DIR] [SEED]
# ---------------------------------------------------------------------------

set -euo pipefail

REAP_DIR="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${REAP_DIR}"
mkdir -p logs

MODEL="${1:?Usage: sbatch slurm_reap_eval.sbatch <MODEL_DIR_OR_HF_ID> [RESULTS_DIR] [SEED]}"
RESULTS_DIR="${2:-}"
SEED="${3:-42}"

# Datasets/models were pre-cached on the login node; compute nodes are offline.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: max-model-len must exceed the eval completion budget (EvalPlus
# MAX_NEW_TOKENS=16384 and the math phase max_new_tokens=16384) PLUS the prompt,
# otherwise vLLM returns 400 "maximum context length" for every request (16384
# completion == 16384 context leaves 0 room for the prompt). 32768 gives 16384
# completion + up to 16384 prompt headroom; KV fits easily on 4x80GB for this
# 3B-active MoE at gpu_mem_util 0.70.
export VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.70}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}

source "${REAP_DIR}/.venv/bin/activate"
set -a; source "${REAP_DIR}/.env"; set +a

# Derive a default results dir from the model path if none was given.
if [[ -z "${RESULTS_DIR}" ]]; then
    SHORT="$(basename "${MODEL}")"
    RESULTS_DIR="${REAP_DIR}/artifacts/eval/${SHORT}"
fi
mkdir -p "${RESULTS_DIR}"

# Pick a port offset from the SLURM job id to avoid collisions.
PORT=$(( 8000 + (${SLURM_JOB_ID:-0} % 1000) ))
SERVER_LOG="${RESULTS_DIR}/vllm_server_${PORT}.log"

echo "=================================================================="
echo "Node:         $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "Model:        ${MODEL}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Seed:         ${SEED}"
echo "vLLM port:    ${PORT}"
echo "=================================================================="

python src/reap/eval.py \
    --model-name "${MODEL}" \
    --results_dir "${RESULTS_DIR}" \
    --seed "${SEED}" \
    --vllm_port "${PORT}" \
    --server_log_file_name "${SERVER_LOG}" \
    --use_server true \
    --run-lm-eval "${RUN_LM_EVAL:-true}" \
    --run-evalplus "${RUN_EVALPLUS:-true}" \
    --run-math "${RUN_MATH:-true}" \
    --run-livecodebench "${RUN_LIVECODEBENCH:-false}" \
    --run-wildbench "${RUN_WILDBENCH:-false}"

echo "=================================================================="
echo "Eval finished. Results under: ${RESULTS_DIR}"
echo "  lm-eval:  ${RESULTS_DIR}/lm_eval_results.json (+ _table.txt)"
echo "  evalplus: ${RESULTS_DIR}/evalplus_results/"
echo "  math:     ${RESULTS_DIR}/evalscope_results/"
echo "=================================================================="
