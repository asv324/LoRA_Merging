#!/bin/bash
# Run the Week 3 Step 3.1 sweep in low-storage mode.
# Each configuration is merged, evaluated, written to its own JSON, then the
# merged adapter directory is deleted to avoid accumulating tens of GB.
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
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/hp_sweep_low_storage}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/hp_sweep_tmp}"

DELTA_LAMBDAS="${DELTA_LAMBDAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
FACTOR_LAMBDAS="${FACTOR_LAMBDAS:-0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0}"
TRIM_PERCENTAGES="${TRIM_PERCENTAGES:-10,20,30}"
DARE_DROP_PROBABILITIES="${DARE_DROP_PROBABILITIES:-0.0,0.1,0.5,0.9}"
DARE_SEED="${DARE_SEED:-42}"
MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPA_VARIANTS_MODE="${GPA_VARIANTS_MODE:-full}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "${RESULTS_ROOT}" "${MERGED_ROOT}"
mkdir -p "${EVAL_CACHE_ROOT}"

export LORA_MERGE_EVAL_CACHE_DIR="${EVAL_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${EVAL_CACHE_ROOT}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${EVAL_CACHE_ROOT}/datasets}"
export HF_EVALUATE_CACHE="${HF_EVALUATE_CACHE:-${EVAL_CACHE_ROOT}/metrics}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${EVAL_CACHE_ROOT}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${EVAL_CACHE_ROOT}/hub}"

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    export LORA_MERGE_LOCAL_FILES_ONLY=1
fi

split_csv() {
    local csv="$1"
    local -n target_array=$2
    IFS=',' read -r -a target_array <<< "${csv}"
}

split_csv "${DELTA_LAMBDAS}" DELTA_LAMBDA_ARRAY
split_csv "${FACTOR_LAMBDAS}" FACTOR_LAMBDA_ARRAY
split_csv "${TRIM_PERCENTAGES}" TRIM_ARRAY
split_csv "${DARE_DROP_PROBABILITIES}" DARE_DROP_ARRAY

format_value() {
    "${PYTHON_CMD}" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
text = f"{value:.4f}".rstrip("0").rstrip(".")
print(text.replace(".", "p"))
PY
}

print_banner() {
    echo
    echo "Week 3 Step 3.1 low-storage sweep"
    echo "Python: ${PYTHON_CMD}"
    echo "Tasks: ${TASKS}"
    echo "Adapters dir: ${ADAPTERS_DIR}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Temporary merged root: ${MERGED_ROOT}"
    echo "Evaluation cache root: ${EVAL_CACHE_ROOT}"
    echo "Delete merged adapters after eval: ${DELETE_AFTER_EVAL}"
    echo "Skip existing: ${SKIP_EXISTING}"
    echo "GPA variants mode: ${GPA_VARIANTS_MODE}"
    echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
    echo "Local files only: ${LOCAL_FILES_ONLY}"
}

run_python() {
    echo
    printf '>>> '
    printf '%q ' "${PYTHON_CMD}" "$@"
    echo
    "${PYTHON_CMD}" "$@"
}

append_eval_args() {
    local -n target_args=$1
    target_args+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
    if [[ -n "${MAX_EVAL_SAMPLES}" ]]; then
        target_args+=(--max-eval-samples "${MAX_EVAL_SAMPLES}")
    fi
}

cleanup_dir() {
    local target_dir="$1"
    if [[ "${DELETE_AFTER_EVAL}" == "1" && -d "${target_dir}" ]]; then
        rm -rf "${target_dir}"
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

task_arithmetic_sweep() {
    local method_root="${MERGED_ROOT}/task_arithmetic"
    local results_dir="${RESULTS_ROOT}/task_arithmetic"
    mkdir -p "${results_dir}"

    for lambda_value in "${DELTA_LAMBDA_ARRAY[@]}"; do
        local lambda_tag
        lambda_tag="$(format_value "${lambda_value}")"
        local adapter_dir="${method_root}/lambda_${lambda_tag}"
        local results_path="${results_dir}/lambda_${lambda_tag}.json"
        local -a cmd=(
            "${PROJECT_ROOT}/scripts/merge_task_arithmetic.py"
            --adapters-dir "${ADAPTERS_DIR}"
            --tasks "${TASK_ARRAY[@]}"
            --lambdas "${lambda_value}"
            --output-dir "${method_root}"
            --results-path "${results_path}"
        )
        append_eval_args cmd
        run_config "Task Arithmetic lambda=${lambda_value}" "${results_path}" "${adapter_dir}" "${cmd[@]}"
    done
}

ties_sweep() {
    local method_root="${MERGED_ROOT}/ties"
    local results_dir="${RESULTS_ROOT}/ties"
    mkdir -p "${results_dir}"

    for trim in "${TRIM_ARRAY[@]}"; do
        local trim_dir="${results_dir}/trim_${trim}"
        mkdir -p "${trim_dir}"
        for lambda_value in "${DELTA_LAMBDA_ARRAY[@]}"; do
            local lambda_tag
            lambda_tag="$(format_value "${lambda_value}")"
            local adapter_dir="${method_root}/trim_${trim}/lambda_${lambda_tag}"
            local results_path="${trim_dir}/lambda_${lambda_tag}.json"
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_ties.py"
                --adapters-dir "${ADAPTERS_DIR}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${method_root}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "TIES trim=${trim} lambda=${lambda_value}" "${results_path}" "${adapter_dir}" "${cmd[@]}"
        done
    done
}

dare_ties_sweep() {
    local method_root="${MERGED_ROOT}/dare"
    local results_dir="${RESULTS_ROOT}/dare_ties"
    mkdir -p "${results_dir}"

    for drop_probability in "${DARE_DROP_ARRAY[@]}"; do
        local drop_tag
        drop_tag="$(format_value "${drop_probability}")"
        local drop_dir="${results_dir}/drop_${drop_tag}"
        mkdir -p "${drop_dir}"
        for trim in "${TRIM_ARRAY[@]}"; do
            local trim_dir="${drop_dir}/trim_${trim}"
            mkdir -p "${trim_dir}"
            for lambda_value in "${DELTA_LAMBDA_ARRAY[@]}"; do
                local lambda_tag
                lambda_tag="$(format_value "${lambda_value}")"
                local adapter_dir="${method_root}/ties/drop_${drop_tag}/trim_${trim}/lambda_${lambda_tag}"
                local results_path="${trim_dir}/lambda_${lambda_tag}.json"
                local -a cmd=(
                    "${PROJECT_ROOT}/scripts/merge_dare.py"
                    --adapters-dir "${ADAPTERS_DIR}"
                    --tasks "${TASK_ARRAY[@]}"
                    --merge-methods ties
                    --drop-probabilities "${drop_probability}"
                    --trim-percentages "${trim}"
                    --lambdas "${lambda_value}"
                    --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                    --seed "${DARE_SEED}"
                    --output-dir "${method_root}"
                    --results-path "${results_path}"
                )
                append_eval_args cmd
                run_config "DARE+TIES p=${drop_probability} trim=${trim} lambda=${lambda_value}" "${results_path}" "${adapter_dir}" "${cmd[@]}"
            done
        done
    done
}

lr_knots_sweep() {
    local method_root="${MERGED_ROOT}/lr_knots"
    local results_dir="${RESULTS_ROOT}/lr_knots"
    mkdir -p "${results_dir}"

    for trim in "${TRIM_ARRAY[@]}"; do
        local trim_dir="${results_dir}/trim_${trim}"
        mkdir -p "${trim_dir}"
        for lambda_value in "${FACTOR_LAMBDA_ARRAY[@]}"; do
            local lambda_tag
            lambda_tag="$(format_value "${lambda_value}")"
            local adapter_dir="${method_root}/trim_${trim}/lambda_${lambda_tag}"
            local results_path="${trim_dir}/lambda_${lambda_tag}.json"
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_lr_knots.py"
                --adapters-dir "${ADAPTERS_DIR}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${method_root}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "LR-KnOTS trim=${trim} lambda=${lambda_value}" "${results_path}" "${adapter_dir}" "${cmd[@]}"
        done
    done
}

gpa_variant_sweep() {
    local slug="$1"
    shift

    local method_root="${MERGED_ROOT}/gpa_ties/${slug}"
    local results_dir="${RESULTS_ROOT}/gpa_ties/${slug}"
    mkdir -p "${results_dir}"

    for trim in "${TRIM_ARRAY[@]}"; do
        local trim_dir="${results_dir}/trim_${trim}"
        mkdir -p "${trim_dir}"
        for lambda_value in "${FACTOR_LAMBDA_ARRAY[@]}"; do
            local lambda_tag
            lambda_tag="$(format_value "${lambda_value}")"
            local adapter_dir="${method_root}/trim_${trim}/lambda_${lambda_tag}"
            local results_path="${trim_dir}/lambda_${lambda_tag}.json"
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
                --adapters-dir "${ADAPTERS_DIR}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${method_root}"
                --results-path "${results_path}"
                "$@"
            )
            append_eval_args cmd
            run_config "GPA variant ${slug} trim=${trim} lambda=${lambda_value}" "${results_path}" "${adapter_dir}" "${cmd[@]}"
        done
    done
}

write_manifest() {
    export RESULTS_ROOT MERGED_ROOT TASKS DELTA_LAMBDAS FACTOR_LAMBDAS TRIM_PERCENTAGES
    export DARE_DROP_PROBABILITIES DARE_SEED MAJORITY_SIGN_METHOD EVAL_BATCH_SIZE
    export MAX_EVAL_SAMPLES SKIP_EXISTING GPA_VARIANTS_MODE DELETE_AFTER_EVAL
    export MODEL_NAME EVAL_CACHE_ROOT WARMUP_EVAL_CACHE LOCAL_FILES_ONLY

    "${PYTHON_CMD}" - <<'PY'
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
result_files = sorted(str(path.relative_to(results_root)) for path in results_root.rglob("*.json"))

manifest = {
    "step": "week3_step3_1_hyperparameter_sweep_low_storage",
    "tasks": os.environ["TASKS"].split(),
    "results_root": os.environ["RESULTS_ROOT"],
    "temporary_merged_root": os.environ["MERGED_ROOT"],
    "evaluation_cache_root": os.environ["EVAL_CACHE_ROOT"],
    "grids": {
        "delta_lambdas": os.environ["DELTA_LAMBDAS"],
        "factor_lambdas": os.environ["FACTOR_LAMBDAS"],
        "trim_percentages": os.environ["TRIM_PERCENTAGES"],
        "dare_drop_probabilities": os.environ["DARE_DROP_PROBABILITIES"],
    },
    "runtime": {
        "dare_seed": int(os.environ["DARE_SEED"]),
        "majority_sign_method": os.environ["MAJORITY_SIGN_METHOD"],
        "eval_batch_size": int(os.environ["EVAL_BATCH_SIZE"]),
        "max_eval_samples": os.environ["MAX_EVAL_SAMPLES"] or None,
        "skip_existing": os.environ["SKIP_EXISTING"] == "1",
        "gpa_variants_mode": os.environ["GPA_VARIANTS_MODE"],
        "delete_after_eval": os.environ["DELETE_AFTER_EVAL"] == "1",
        "model_name": os.environ["MODEL_NAME"],
        "warmup_eval_cache": os.environ["WARMUP_EVAL_CACHE"] == "1",
        "local_files_only": os.environ["LOCAL_FILES_ONLY"] == "1",
    },
    "result_file_count": len(result_files),
    "result_files": result_files,
}

(results_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved low-storage sweep manifest to {results_root / 'run_manifest.json'}")
PY
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

task_arithmetic_sweep
ties_sweep
dare_ties_sweep
lr_knots_sweep
gpa_variant_sweep baseline

case "${GPA_VARIANTS_MODE}" in
    full)
        gpa_variant_sweep dgpa_ties --normalise-a-factors
        gpa_variant_sweep dgpa_saties --normalise-a-factors --scale-aware-ties
        gpa_variant_sweep dgpa_saties_wb_0p5 --normalise-a-factors --scale-aware-ties --b-weight-alpha 0.5
        gpa_variant_sweep dgpa_saties_wb_1p0 --normalise-a-factors --scale-aware-ties --b-weight-alpha 1.0
        ;;
    minimal)
        gpa_variant_sweep dgpa_saties_wb_1p0 --normalise-a-factors --scale-aware-ties --b-weight-alpha 1.0
        ;;
    none)
        ;;
    *)
        echo "Unsupported GPA_VARIANTS_MODE: ${GPA_VARIANTS_MODE}" >&2
        exit 1
        ;;
esac

write_manifest

echo
echo "Week 3 low-storage sweep complete."
echo "Results: ${RESULTS_ROOT}"
