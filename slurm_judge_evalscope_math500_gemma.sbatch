#!/usr/bin/env bash
#SBATCH --job-name=gemma_math500
#SBATCH --partition=gpumid
#SBATCH --account=cerebras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/scratch/users/biswajit.mishra/model_merge/logs/gemma_math500_%j.out
#SBATCH --error=/lustre/scratch/users/biswajit.mishra/model_merge/logs/gemma_math500_%j.err

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: sbatch $0 <result-dir> [<result-dir> ...]" >&2
  exit 2
fi

ROOT=/lustre/scratch/users/biswajit.mishra/model_merge
export HF_HOME=${ROOT}/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR=${ROOT}/.cache/torchinductor_gemma_math_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${ROOT}/.cache/triton_gemma_math_${SLURM_JOB_ID}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

source /lustre/scratch/users/biswajit.mishra/venv023/bin/activate
python ${ROOT}/msmoe/reap/scripts/judge_evalscope_math500_gemma.py "$@"