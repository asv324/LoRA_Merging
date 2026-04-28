#!/bin/bash
# Week 4 Step 4.1 / Experiment 11 driver: vary the number of merged adapters.
#
# Runs all N in {2, 3, 4} subset configurations for the four Experiment 11
# methods at their best restored-head Week 3 hyperparameters, then aggregates
# the results together with the N = 5 restored-head argmax row from
# results/best_hparams.json into a single summary and emits the headline figure.
#
# This script only sets up environment variables and delegates the heavy
# lifting to scripts/run_ablation_N.py, scripts/analyze_ablation_N.py, and
# scripts/plot_ablation_N.py. Keeping the bash wrapper thin mirrors the
# division of labour used in run_week3_statistical_significance.sh while
# avoiding a second bash implementation of the combinatorial subset logic.
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
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/ablation_N}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/ablation_N_tmp}"
BEST_HPARAMS_PATH="${BEST_HPARAMS_PATH:-${PROJECT_ROOT}/results/best_hparams.json}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/ablation_N}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"

N_VALUES="${N_VALUES:-2,3,4}"
METHODS="${METHODS:-gpa_baseline,gpa_dgpa_saties_wb_0p5,ties,lr_knots}"
MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
KEEP_MERGED="${KEEP_MERGED:-0}"
RUN_MERGES="${RUN_MERGES:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RUN_PLOT="${RUN_PLOT:-1}"

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
    echo "Week 4 Step 4.1 / Experiment 11 (vary N)"
    echo "Python: ${PYTHON_CMD}"
    echo "Adapters dir: ${ADAPTERS_DIR}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Temporary merged root: ${MERGED_ROOT}"
    echo "N=5 source: ${BEST_HPARAMS_PATH}"
    echo "N values to run: ${N_VALUES}"
    echo "Methods: ${METHODS}"
    echo "Eval batch size: ${EVAL_BATCH_SIZE}"
    echo "Max eval samples: ${MAX_EVAL_SAMPLES:-<full>}"
    echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
    echo "Local files only: ${LOCAL_FILES_ONLY}"
    echo "Keep merged adapters: ${KEEP_MERGED}"
    echo "Run merges/analysis/plot: ${RUN_MERGES}/${RUN_ANALYSIS}/${RUN_PLOT}"
}

print_banner

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

if [[ "${RUN_MERGES}" == "1" ]]; then
    merge_args=(
        "${PROJECT_ROOT}/scripts/run_ablation_N.py"
        --python-bin "${PYTHON_CMD}"
        --adapters-dir "${ADAPTERS_DIR}"
        --results-root "${RESULTS_ROOT}"
        --merged-root "${MERGED_ROOT}"
        --best-hparams-path "${BEST_HPARAMS_PATH}"
        --n-values "${N_VALUES}"
        --methods "${METHODS}"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --eval-batch-size "${EVAL_BATCH_SIZE}"
    )
    if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
        merge_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
    fi
    if [[ "${KEEP_MERGED}" == "1" ]]; then
        merge_args+=(--keep-merged)
    fi

    log_path="${LOG_ROOT}/run_merges.log"
    echo
    echo ">>> Running N-ablation merges (log: ${log_path})"
    "${PYTHON_CMD}" "${merge_args[@]}" 2>&1 | tee "${log_path}"
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo
    echo ">>> Aggregating per-(N, method) summary"
    "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/analyze_ablation_N.py" \
        --results-root "${RESULTS_ROOT}" \
        --best-hparams-path "${BEST_HPARAMS_PATH}" \
        --methods "${METHODS}"
fi

if [[ "${RUN_PLOT}" == "1" ]]; then
    echo
    echo ">>> Emitting ablation-N figure"
    "${PYTHON_CMD}" "${PROJECT_ROOT}/scripts/plot_ablation_N.py" \
        --summary-path "${RESULTS_ROOT}/summary.json" \
        --output-path "${PROJECT_ROOT}/dissertation/chapters/figures/ablation_N.pdf"
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
echo "Step 4.1 ablation-N pipeline complete."
echo "Per-config results: ${RESULTS_ROOT}"
echo "Summary: ${RESULTS_ROOT}/summary.json"
echo "Figure: ${PROJECT_ROOT}/dissertation/chapters/figures/ablation_N.pdf"
