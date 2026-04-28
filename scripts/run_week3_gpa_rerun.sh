#!/bin/bash
# Run the targeted Week 3 GPA rerun in the restored-classifier-head regime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
else
    echo "Could not find python. Set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

TASKS="${TASKS:-sst2 mnli qnli cola rte}"
read -r -a TASK_ARRAY <<< "${TASKS}"

ADAPTERS_DIR="${ADAPTERS_DIR:-${PROJECT_ROOT}/adapters}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/hp_sweep_gpa_rerun_restored_heads}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/hp_sweep_gpa_rerun_restored_heads}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
MAX_ITER="${MAX_ITER:-300}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-0}"

mkdir -p "${RESULTS_ROOT}" "${MERGED_ROOT}" "${EVAL_CACHE_ROOT}"

export LORA_MERGE_EVAL_CACHE_DIR="${EVAL_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${EVAL_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EVAL_CACHE_ROOT}/datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${EVAL_CACHE_ROOT}/metrics}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${EVAL_CACHE_ROOT}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${EVAL_CACHE_ROOT}/hub}"

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    export LORA_MERGE_LOCAL_FILES_ONLY=1
fi

print_banner() {
    echo
    echo "Week 3 targeted GPA rerun"
    echo "Python: ${PYTHON_CMD}"
    echo "Tasks: ${TASKS}"
    echo "Adapters dir: ${ADAPTERS_DIR}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Merged root: ${MERGED_ROOT}"
    echo "Evaluation cache root: ${EVAL_CACHE_ROOT}"
    echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
    echo "Local files only: ${LOCAL_FILES_ONLY}"
    echo "Max iter: ${MAX_ITER}"
    echo "Skip existing: ${SKIP_EXISTING}"
    echo "Delete merged adapters after eval: ${DELETE_AFTER_EVAL}"
}

run_python() {
    echo
    printf '>>> '
    printf '%q ' "${PYTHON_CMD}" "$@"
    echo
    "${PYTHON_CMD}" "$@"
}

cleanup_dir() {
    local target_dir="$1"
    if [[ "${DELETE_AFTER_EVAL}" == "1" && -d "${target_dir}" ]]; then
        rm -rf "${target_dir}"
    fi
}

append_eval_args() {
    local -n target_args=$1
    target_args+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
    if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
        target_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
    fi
}

run_config() {
    local label="$1"
    local results_path="$2"
    local adapter_dir="$3"
    shift 3

    if [[ "${SKIP_EXISTING}" == "1" && -f "${results_path}" ]]; then
        echo
        echo ">>> Skipping ${label}; found existing results at ${results_path}"
        return
    fi

    run_python "$@"
    if [[ ! -f "${results_path}" ]]; then
        echo "Expected results file was not created: ${results_path}" >&2
        exit 1
    fi
    cleanup_dir "${adapter_dir}"
}

run_baseline() {
    local method_root="${MERGED_ROOT}/gpa_ties/baseline"
    local results_path="${RESULTS_ROOT}/gpa_ties/baseline/trim_30/lambda_1.json"
    local adapter_dir="${method_root}/trim_30/lambda_1"
    mkdir -p "$(dirname "${results_path}")"

    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "30"
        --lambdas "1.0"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --max-iter "${MAX_ITER}"
        --output-dir "${method_root}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    run_config "GPA baseline rerun" "${results_path}" "${adapter_dir}" "${cmd[@]}"
}

run_dgpa_saties_wb_0p5() {
    local method_root="${MERGED_ROOT}/gpa_ties/dgpa_saties_wb_0p5"
    local results_path="${RESULTS_ROOT}/gpa_ties/dgpa_saties_wb_0p5/trim_30/lambda_1.json"
    local adapter_dir="${method_root}/trim_30/lambda_1"
    mkdir -p "$(dirname "${results_path}")"

    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "30"
        --lambdas "1.0"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --normalise-a-factors
        --scale-aware-ties
        --b-weight-alpha "0.5"
        --max-iter "${MAX_ITER}"
        --output-dir "${method_root}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    run_config "dGPA+saTIES+wB(0.5) rerun" "${results_path}" "${adapter_dir}" "${cmd[@]}"
}

run_dgpa_saties_wb_1p0() {
    local method_root="${MERGED_ROOT}/gpa_ties/dgpa_saties_wb_1p0"
    local results_path="${RESULTS_ROOT}/gpa_ties/dgpa_saties_wb_1p0/trim_30/lambda_1.json"
    local adapter_dir="${method_root}/trim_30/lambda_1"
    mkdir -p "$(dirname "${results_path}")"

    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "30"
        --lambdas "1.0"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --normalise-a-factors
        --scale-aware-ties
        --b-weight-alpha "1.0"
        --max-iter "${MAX_ITER}"
        --output-dir "${method_root}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    run_config "dGPA+saTIES+wB(1.0) rerun" "${results_path}" "${adapter_dir}" "${cmd[@]}"
}

collect_cola_summary() {
    local output_path="${RESULTS_ROOT}/cola_prediction_distributions.json"
    run_python \
        "${PROJECT_ROOT}/scripts/collect_cola_prediction_distributions.py" \
        --sweep-root "${RESULTS_ROOT}" \
        --output-path "${output_path}"
}

print_banner

if [[ "${WARMUP_EVAL_CACHE}" == "1" ]]; then
    local_only_flag=()
    if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
        local_only_flag+=(--local-files-only)
    fi
    run_python \
        "${PROJECT_ROOT}/scripts/warmup_eval_cache.py" \
        --model-name "${MODEL_NAME}" \
        --tasks "${TASK_ARRAY[@]}" \
        --max-length 256 \
        "${local_only_flag[@]}"
fi

run_baseline
run_dgpa_saties_wb_0p5
run_dgpa_saties_wb_1p0
collect_cola_summary

echo
echo "Week 3 targeted GPA rerun complete."
echo "Results: ${RESULTS_ROOT}"
