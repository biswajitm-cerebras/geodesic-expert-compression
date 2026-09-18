#!/bin/bash
# Controlled comparison on Qwen3-30B-A3B-Instruct-2507:
# dense versus 50% REAP (seeds 42, 11, 99).
set -euo pipefail

ROOT="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${ROOT}"
mkdir -p logs artifacts/eval/instruct2507_controlled

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_ROOT="${ROOT}/artifacts/eval/instruct2507_controlled"

submit_eval() {
    local dependency="$1" model="$2" output="$3" phase="$4" seed="$5"
    local flags
    case "${phase}" in
        core)
            flags="RUN_LM_EVAL=true,RUN_EVALPLUS=true,RUN_LIVECODEBENCH=true,RUN_MATH=false,RUN_WILDBENCH=false"
            ;;
        gsm8k)
            flags="RUN_LM_EVAL=false,RUN_EVALPLUS=false,RUN_LIVECODEBENCH=false,RUN_MATH=true,RUN_WILDBENCH=false,MATH_DATASETS=gsm8k"
            ;;
        math500)
            flags="RUN_LM_EVAL=false,RUN_EVALPLUS=false,RUN_LIVECODEBENCH=false,RUN_MATH=true,RUN_WILDBENCH=false,MATH_DATASETS=math_500"
            ;;
        *)
            echo "Unknown phase: ${phase}" >&2
            return 2
            ;;
    esac

    local args=(--parsable --job-name="i2507_${phase}" --export="ALL,${flags}")
    if [[ -n "${dependency}" ]]; then
        args+=(--dependency="afterok:${dependency}")
    fi
    sbatch "${args[@]}" slurm_reap_eval.sbatch "${model}" "${output}/${phase}" "${seed}"
}

# Dense is deterministic under greedy decoding, so one run is sufficient.
DENSE_OUT="${OUTPUT_ROOT}/dense"
DENSE_CORE=$(submit_eval "" "${MODEL}" "${DENSE_OUT}" core 42)
DENSE_GSM=$(submit_eval "" "${MODEL}" "${DENSE_OUT}" gsm8k 42)
DENSE_MATH=$(submit_eval "" "${MODEL}" "${DENSE_OUT}" math500 42)

# REAP saliency depends on calibration shuffling, so reproduce three seeds.
PRUNE_ARRAY=$(sbatch --parsable --array=0-2 slurm_instruct2507_reap_prune.sbatch)
PRUNE_JOB="${PRUNE_ARRAY%%;*}"
SEEDS=(42 11 99)
REAP_JOBS=()
for index in 0 1 2; do
    seed="${SEEDS[$index]}"
    dependency="${PRUNE_JOB}_${index}"
    checkpoint="${ROOT}/artifacts/Qwen3-30B-A3B-Instruct-2507/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_${seed}-0.50"
    output="${OUTPUT_ROOT}/reap_seed_${seed}"
    REAP_JOBS+=("$(submit_eval "${dependency}" "${checkpoint}" "${output}" core "${seed}")")
    REAP_JOBS+=("$(submit_eval "${dependency}" "${checkpoint}" "${output}" gsm8k "${seed}")")
    REAP_JOBS+=("$(submit_eval "${dependency}" "${checkpoint}" "${output}" math500 "${seed}")")
done

MANIFEST="${OUTPUT_ROOT}/jobs.txt"
{
    echo "submitted=$(date --iso-8601=seconds)"
    echo "model=${MODEL}"
    echo "dense_core=${DENSE_CORE}"
    echo "dense_gsm8k=${DENSE_GSM}"
    echo "dense_math500=${DENSE_MATH}"
    echo "reap_prune_array=${PRUNE_ARRAY}"
    printf 'reap_eval=%s\n' "${REAP_JOBS[@]}"
} | tee "${MANIFEST}"

echo "Manifest: ${MANIFEST}"
