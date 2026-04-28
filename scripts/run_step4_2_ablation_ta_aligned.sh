#!/bin/bash
# Week 4 Step 4.2 / Experiment 12 driver: Task Arithmetic in aligned space.
#
# Runs the two methodology-declared rows at the restored-head best lambda for
# raw Task Arithmetic from results/best_hparams.json (overridable via
# LAMBDA_VALUE=...):
#
# 1. GPA-aligned TA:          plain GPA alignment + unweighted aligned TA.
#                              Delta-preserving; numerically matches the
#                              unaligned TA reference row up to float noise.
# 2. enhanced-GPA-aligned TA: directional dGPA alignment + inverse-norm B
#                              weighting (alpha=0.5) feeding into aligned TA.
#                              Breaks delta-preservation; this is the
#                              substantive Experiment 12 row.
#
# After both runs complete, the analysis step emits a single
# results/ablation_ta_aligned/summary.json with the two rows plus their delta
# versus the unaligned Task Arithmetic row loaded from results/best_hparams.json.
#
# Budgeted at ~3h in Documentation/revised_implementation_plan_v2.md.
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

ADAPTERS_DIR="${ADAPTERS_DIR:-${PROJECT_ROOT}/adapters}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/ablation_ta_aligned}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/ablation_ta_aligned_tmp}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/ablation_ta_aligned}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"
BEST_HPARAMS_PATH="${BEST_HPARAMS_PATH:-${PROJECT_ROOT}/results/best_hparams.json}"
B_WEIGHT_ALPHA_ENHANCED="${B_WEIGHT_ALPHA_ENHANCED:-0.5}"
TASKS="${TASKS:-sst2 mnli qnli cola rte}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
MAX_ITER="${MAX_ITER:-100}"
INIT="${INIT:-first}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
KEEP_MERGED="${KEEP_MERGED:-0}"
RUN_MERGES="${RUN_MERGES:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
VARIANTS="${VARIANTS:-gpa_aligned_ta,enhanced_gpa_aligned_ta}"

DEFAULT_TA_LAMBDA="$("${PYTHON_CMD}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["methods"]["task_arithmetic"]["hyperparameters"]["lambda"])' "${BEST_HPARAMS_PATH}")"
LAMBDA_VALUE="${LAMBDA_VALUE:-${DEFAULT_TA_LAMBDA}}"

mkdir -p "${RESULTS_ROOT}" "${MERGED_ROOT}" "${LOG_ROOT}" "${EVAL_CACHE_ROOT}"

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
    echo "Week 4 Step 4.2 / Experiment 12 (Task Arithmetic in aligned space)"
    echo "Python: ${PYTHON_CMD}"
    echo "Adapters dir: ${ADAPTERS_DIR}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Temporary merged root: ${MERGED_ROOT}"
    echo "Lambda: ${LAMBDA_VALUE}"
    echo "Enhanced b_weight_alpha: ${B_WEIGHT_ALPHA_ENHANCED}"
    echo "Tasks: ${TASKS}"
    echo "Eval batch size: ${EVAL_BATCH_SIZE}"
    echo "Max eval samples: ${MAX_EVAL_SAMPLES:-<full>}"
    echo "GPA max_iter: ${MAX_ITER}"
    echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
    echo "Local files only: ${LOCAL_FILES_ONLY}"
    echo "Keep merged adapters: ${KEEP_MERGED}"
    echo "Run merges/analysis: ${RUN_MERGES}/${RUN_ANALYSIS}"
    echo "Variants: ${VARIANTS}"
}

print_banner

read -r -a TASK_ARRAY <<< "${TASKS}"

if [[ "${WARMUP_EVAL_CACHE}" == "1" && "${RUN_MERGES}" == "1" ]]; then
    local_only_flag=()
    if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
        local_only_flag+=(--local-files-only)
    fi
    echo
    echo ">>> Warming HF caches for all five GLUE tasks"
    "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/warmup_eval_cache.py" \
        --model-name "${MODEL_NAME}" \
        --tasks sst2 mnli qnli cola rte \
        --max-length 256 \
        "${local_only_flag[@]}"
fi

run_variant() {
    local alias="$1"
    shift
    local extra_args=("$@")

    local results_path="${RESULTS_ROOT}/${alias}.json"
    local merged_dir="${MERGED_ROOT}/${alias}"
    local log_path="${LOG_ROOT}/${alias}.log"

    echo
    echo ">>> Running variant '${alias}' (log: ${log_path})"

    local cmd=(
        "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/merge_ta_aligned.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --lambdas "${LAMBDA_VALUE}"
        --max-iter "${MAX_ITER}"
        --init "${INIT}"
        --variant-label "${alias}"
        --output-dir "${merged_dir}"
        --results-path "${results_path}"
        --eval-batch-size "${EVAL_BATCH_SIZE}"
    )
    if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
        cmd+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
    fi
    cmd+=("${extra_args[@]}")

    "${cmd[@]}" 2>&1 | tee "${log_path}"

    if [[ "${KEEP_MERGED}" != "1" && -d "${merged_dir}" ]]; then
        rm -rf "${merged_dir}"
    fi
}

if [[ "${RUN_MERGES}" == "1" ]]; then
    IFS=',' read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
    for alias in "${VARIANT_ARRAY[@]}"; do
        alias="${alias// /}"
        case "${alias}" in
            gpa_aligned_ta)
                run_variant "${alias}"
                ;;
            enhanced_gpa_aligned_ta)
                run_variant "${alias}" \
                    --normalise-a-factors \
                    --b-weight-alpha "${B_WEIGHT_ALPHA_ENHANCED}"
                ;;
            *)
                echo "Unknown variant alias '${alias}'. Supported: gpa_aligned_ta, enhanced_gpa_aligned_ta." >&2
                exit 2
                ;;
        esac
    done
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo
    echo ">>> Aggregating Experiment 12 summary"
    "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/analyze_ablation_ta_aligned.py" \
        --results-root "${RESULTS_ROOT}" \
        --variants "${VARIANTS}" \
        --lambda "${LAMBDA_VALUE}" \
        --best-hparams-path "${BEST_HPARAMS_PATH}"
fi

cleanup_merged_root() {
    if [[ "${KEEP_MERGED}" == "1" ]]; then
        return
    fi
    if [[ ! -d "${MERGED_ROOT}" ]]; then
        return
    fi
    find "${MERGED_ROOT}" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
    if [[ -d "${MERGED_ROOT}" ]] && [[ -z "$(ls -A "${MERGED_ROOT}" 2>/dev/null)" ]]; then
        rmdir "${MERGED_ROOT}" 2>/dev/null || true
    fi
}

cleanup_merged_root

echo
echo "Step 4.2 aligned-TA pipeline complete."
echo "Per-variant results: ${RESULTS_ROOT}"
echo "Summary: ${RESULTS_ROOT}/summary.json"
