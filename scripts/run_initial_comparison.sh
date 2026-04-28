#!/bin/bash
# Run the Week 2 Step 2.8 initial comparison in the restored-classifier-head
# evaluation regime.
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

LAMBDA_VALUE="${LAMBDA_VALUE:-1.0}"
TRIM_PERCENTAGE="${TRIM_PERCENTAGE:-20}"
DROP_PROBABILITY="${DROP_PROBABILITY:-0.1}"
MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
TASKS="${TASKS:-sst2 mnli qnli cola rte}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

MERGED_ROOT="${PROJECT_ROOT}/merged_adapters/initial_comparison_restored_heads"
RESULTS_ROOT="${PROJECT_ROOT}/results/initial_comparison_restored_heads"
mkdir -p "${MERGED_ROOT}" "${RESULTS_ROOT}" "${EVAL_CACHE_ROOT}"

export LORA_MERGE_EVAL_CACHE_DIR="${EVAL_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${EVAL_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EVAL_CACHE_ROOT}/datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${EVAL_CACHE_ROOT}/metrics}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${EVAL_CACHE_ROOT}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${EVAL_CACHE_ROOT}/hub}"

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    export LORA_MERGE_LOCAL_FILES_ONLY=1
fi

echo "Using Python: ${PYTHON_CMD}"
echo "Tasks: ${TASKS}"
echo "Lambda: ${LAMBDA_VALUE}"
echo "Trim percentage: ${TRIM_PERCENTAGE}"
echo "Drop probability: ${DROP_PROBABILITY}"
echo "Evaluation cache root: ${EVAL_CACHE_ROOT}"
echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
echo "Local files only: ${LOCAL_FILES_ONLY}"

run_python() {
    echo
    echo ">>> $*"
    "${PYTHON_CMD}" "$@"
}

format_value() {
    "${PYTHON_CMD}" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
text = f"{value:.4f}".rstrip("0").rstrip(".")
print(text.replace(".", "p"))
PY
}

maybe_add_max_eval_samples() {
    if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
        printf -- ' --max-eval-samples %s' "${MAX_EVAL_SAMPLES}"
    fi
}

LAMBDA_TAG="$(format_value "${LAMBDA_VALUE}")"
DROP_TAG="$(format_value "${DROP_PROBABILITY}")"

if [[ "${WARMUP_EVAL_CACHE}" == "1" ]]; then
    local_only_flag=()
    if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
        local_only_flag+=(--local-files-only)
    fi
    run_python "${PROJECT_ROOT}/scripts/warmup_eval_cache.py" \
        --model-name "${MODEL_NAME}" \
        --tasks ${TASKS} \
        --max-length 256 \
        "${local_only_flag[@]}"
fi

run_python "${PROJECT_ROOT}/scripts/merge_task_arithmetic.py" \
    --tasks ${TASKS} \
    --lambdas "${LAMBDA_VALUE}" \
    --skip-eval \
    --output-dir "${MERGED_ROOT}/task_arithmetic" \
    --results-path "${RESULTS_ROOT}/task_arithmetic_merge.json"

run_python "${PROJECT_ROOT}/scripts/merge_ties.py" \
    --tasks ${TASKS} \
    --trim-percentages "${TRIM_PERCENTAGE}" \
    --lambdas "${LAMBDA_VALUE}" \
    --majority-sign-method "${MAJORITY_SIGN_METHOD}" \
    --skip-eval \
    --output-dir "${MERGED_ROOT}/ties" \
    --results-path "${RESULTS_ROOT}/ties_merge.json"

run_python "${PROJECT_ROOT}/scripts/merge_dare.py" \
    --tasks ${TASKS} \
    --merge-methods ties \
    --drop-probabilities "${DROP_PROBABILITY}" \
    --trim-percentages "${TRIM_PERCENTAGE}" \
    --lambdas "${LAMBDA_VALUE}" \
    --majority-sign-method "${MAJORITY_SIGN_METHOD}" \
    --skip-eval \
    --output-dir "${MERGED_ROOT}/dare" \
    --results-path "${RESULTS_ROOT}/dare_merge.json"

run_python "${PROJECT_ROOT}/scripts/merge_lr_knots.py" \
    --tasks ${TASKS} \
    --trim-percentages "${TRIM_PERCENTAGE}" \
    --lambdas "${LAMBDA_VALUE}" \
    --majority-sign-method "${MAJORITY_SIGN_METHOD}" \
    --skip-eval \
    --output-dir "${MERGED_ROOT}/lr_knots" \
    --results-path "${RESULTS_ROOT}/lr_knots_merge.json"

run_python "${PROJECT_ROOT}/scripts/merge_gpa_ties.py" \
    --tasks ${TASKS} \
    --trim-percentages "${TRIM_PERCENTAGE}" \
    --lambdas "${LAMBDA_VALUE}" \
    --majority-sign-method "${MAJORITY_SIGN_METHOD}" \
    --skip-eval \
    --output-dir "${MERGED_ROOT}/gpa_ties" \
    --results-path "${RESULTS_ROOT}/gpa_ties_merge.json"

EVAL_MAX_SAMPLES_ARGS="$(maybe_add_max_eval_samples)"

run_python "${PROJECT_ROOT}/scripts/evaluate_merged.py" \
    --adapter-dir "${MERGED_ROOT}/task_arithmetic/lambda_${LAMBDA_TAG}" \
    --tasks ${TASKS} \
    --batch-size "${EVAL_BATCH_SIZE}" \
    ${EVAL_MAX_SAMPLES_ARGS} \
    --output-path "${RESULTS_ROOT}/task_arithmetic_eval.json"

run_python "${PROJECT_ROOT}/scripts/evaluate_merged.py" \
    --adapter-dir "${MERGED_ROOT}/ties/trim_${TRIM_PERCENTAGE}/lambda_${LAMBDA_TAG}" \
    --tasks ${TASKS} \
    --batch-size "${EVAL_BATCH_SIZE}" \
    ${EVAL_MAX_SAMPLES_ARGS} \
    --output-path "${RESULTS_ROOT}/ties_eval.json"

run_python "${PROJECT_ROOT}/scripts/evaluate_merged.py" \
    --adapter-dir "${MERGED_ROOT}/dare/ties/drop_${DROP_TAG}/trim_${TRIM_PERCENTAGE}/lambda_${LAMBDA_TAG}" \
    --tasks ${TASKS} \
    --batch-size "${EVAL_BATCH_SIZE}" \
    ${EVAL_MAX_SAMPLES_ARGS} \
    --output-path "${RESULTS_ROOT}/dare_ties_eval.json"

run_python "${PROJECT_ROOT}/scripts/evaluate_merged.py" \
    --adapter-dir "${MERGED_ROOT}/lr_knots/trim_${TRIM_PERCENTAGE}/lambda_${LAMBDA_TAG}" \
    --tasks ${TASKS} \
    --batch-size "${EVAL_BATCH_SIZE}" \
    ${EVAL_MAX_SAMPLES_ARGS} \
    --output-path "${RESULTS_ROOT}/lr_knots_eval.json"

run_python "${PROJECT_ROOT}/scripts/evaluate_merged.py" \
    --adapter-dir "${MERGED_ROOT}/gpa_ties/trim_${TRIM_PERCENTAGE}/lambda_${LAMBDA_TAG}" \
    --tasks ${TASKS} \
    --batch-size "${EVAL_BATCH_SIZE}" \
    ${EVAL_MAX_SAMPLES_ARGS} \
    --output-path "${RESULTS_ROOT}/gpa_ties_eval.json"

export PROJECT_ROOT RESULTS_ROOT LAMBDA_VALUE TRIM_PERCENTAGE DROP_PROBABILITY MAJORITY_SIGN_METHOD
"${PYTHON_CMD}" - <<'PY'
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
trim_percentage = int(float(os.environ["TRIM_PERCENTAGE"]))
lambda_value = float(os.environ["LAMBDA_VALUE"])
drop_probability = float(os.environ["DROP_PROBABILITY"])
majority_sign_method = os.environ["MAJORITY_SIGN_METHOD"]


def load_json(name: str):
    return json.loads((results_root / name).read_text(encoding="utf-8"))


def method_block(eval_name: str, *, label: str, trim: int | None = None, drop: float | None = None):
    payload = load_json(eval_name)
    run = payload["runs"][0]
    block = {
        "label": label,
        "adapter_dir": payload["adapter_dir"],
        "lambda": run["lambda"],
        "per_task_primary_metrics": run["summary"]["primary_metrics"],
        "average_primary_score": run["summary"]["average_primary_score"],
    }
    if trim is not None:
        block["trim_percentage"] = trim
        block["majority_sign_method"] = majority_sign_method
    if drop is not None:
        block["drop_probability"] = drop
    return block


comparison = {
    "task_arithmetic": method_block("task_arithmetic_eval.json", label="Task Arithmetic"),
    "ties": method_block("ties_eval.json", label="TIES-Merging", trim=trim_percentage),
    "dare_ties": method_block("dare_ties_eval.json", label="DARE + TIES", trim=trim_percentage, drop=drop_probability),
    "lr_knots_ties": method_block("lr_knots_eval.json", label="LR-KnOTS + TIES", trim=trim_percentage),
    "gpa_ties": method_block("gpa_ties_eval.json", label="GPA + TIES", trim=trim_percentage),
    "comparison_settings": {
        "lambda": lambda_value,
        "trim_percentage": trim_percentage,
        "drop_probability": drop_probability,
        "majority_sign_method": majority_sign_method,
    },
}

output_path = results_root / "initial_comparison.json"
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(f"Saved initial comparison summary to {output_path}")
PY

echo
echo "Initial comparison complete."
echo "Summary: ${RESULTS_ROOT}/initial_comparison.json"
