#!/bin/bash
# Reproduce the Qwen3-30B-A3B dense and 50%-REAP paper benchmarks.
set -euo pipefail

ROOT="/lustre/scratch/users/biswajit.mishra/model_merge/msmoe/reap"
cd "${ROOT}"
mkdir -p logs artifacts/eval/paper_reproduction

DENSE_MODEL="Qwen/Qwen3-30B-A3B"
DENSE_OUT="${ROOT}/artifacts/eval/paper_reproduction/dense"

submit_eval() {
    local dependency="$1" model="$2" output="$3" phase="$4"
    local export_flags
    case "${phase}" in
        core)
            export_flags="RUN_LM_EVAL=true,RUN_EVALPLUS=true,RUN_LIVECODEBENCH=true,RUN_MATH=false,RUN_WILDBENCH=false"
            ;;
        gsm8k)
            export_flags="RUN_LM_EVAL=false,RUN_EVALPLUS=false,RUN_LIVECODEBENCH=false,RUN_MATH=true,RUN_WILDBENCH=false,MATH_DATASETS=gsm8k"
            ;;
        math500)
            export_flags="RUN_LM_EVAL=false,RUN_EVALPLUS=false,RUN_LIVECODEBENCH=false,RUN_MATH=true,RUN_WILDBENCH=false,MATH_DATASETS=math_500"
            ;;
        *)
            echo "Unknown phase: ${phase}" >&2
            return 2
            ;;
    esac

    local args=(--parsable --job-name="paper_${phase}" --export="ALL,${export_flags}")
    if [[ -n "${dependency}" ]]; then
        args+=(--dependency="afterok:${dependency}")
    fi
    sbatch "${args[@]}" slurm_reap_eval.sbatch "${model}" "${output}/${phase}" 42
}

# Dense checkpoint: one deterministic evaluation per benchmark family.
DENSE_CORE=$(submit_eval "" "${DENSE_MODEL}" "${DENSE_OUT}" core)
DENSE_GSM=$(submit_eval "" "${DENSE_MODEL}" "${DENSE_OUT}" gsm8k)
DENSE_MATH=$(submit_eval "" "${DENSE_MODEL}" "${DENSE_OUT}" math500)

# Paper REAP reports a three-seed mean. Build all three checkpoints as an array.
PRUNE_ARRAY=$(sbatch --parsable --array=0-2 slurm_paper_reap_prune.sbatch)
PRUNE_JOB="${PRUNE_ARRAY%%;*}"

SEEDS=(42 11 99)
REAP_JOBS=()
for index in 0 1 2; do
    seed="${SEEDS[$index]}"
    dependency="${PRUNE_JOB}_${index}"
    model="${ROOT}/artifacts/Qwen3-30B-A3B/evol-codealpaca-v1/pruned_models/reap-renorm_true-seed_${seed}-0.50"
    output="${ROOT}/artifacts/eval/paper_reproduction/reap_seed_${seed}"
    REAP_JOBS+=("$(submit_eval "${dependency}" "${model}" "${output}" core)")
    REAP_JOBS+=("$(submit_eval "${dependency}" "${model}" "${output}" gsm8k)")
    REAP_JOBS+=("$(submit_eval "${dependency}" "${model}" "${output}" math500)")
done

MANIFEST="${ROOT}/artifacts/eval/paper_reproduction/jobs.txt"
{
    echo "submitted=$(date --iso-8601=seconds)"
    echo "dense_model=${DENSE_MODEL}"
    echo "dense_core=${DENSE_CORE}"
    echo "dense_gsm8k=${DENSE_GSM}"
    echo "dense_math500=${DENSE_MATH}"
    echo "reap_prune_array=${PRUNE_ARRAY}"
    printf 'reap_eval=%s\n' "${REAP_JOBS[@]}"
} | tee "${MANIFEST}"

echo "Manifest: ${MANIFEST}"
