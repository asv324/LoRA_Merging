#!/bin/bash
# Run the Week 3 Step 3.1 hyperparameter sweep sequentially in the
# restored-classifier-head evaluation regime.
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
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/hp_sweep_restored_heads}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/hp_sweep_restored_heads}"

DELTA_LAMBDAS="${DELTA_LAMBDAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
FACTOR_LAMBDAS="${FACTOR_LAMBDAS:-0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.7,1.0}"
TRIM_PERCENTAGES="${TRIM_PERCENTAGES:-10,20,30}"
DARE_DROP_PROBABILITIES="${DARE_DROP_PROBABILITIES:-0.0,0.1,0.5,0.9}"
DARE_SEED="${DARE_SEED:-42}"
MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPA_VARIANTS_MODE="${GPA_VARIANTS_MODE:-full}"
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

print_banner() {
    echo
    echo "Week 3 Step 3.1 hyperparameter sweep"
    echo "Python: ${PYTHON_CMD}"
    echo "Tasks: ${TASKS}"
    echo "Adapters dir: ${ADAPTERS_DIR}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Merged adapters root: ${MERGED_ROOT}"
    echo "Evaluation cache root: ${EVAL_CACHE_ROOT}"
    echo "Delta lambdas: ${DELTA_LAMBDAS}"
    echo "Factor lambdas: ${FACTOR_LAMBDAS}"
    echo "Trim percentages: ${TRIM_PERCENTAGES}"
    echo "DARE drop probabilities: ${DARE_DROP_PROBABILITIES}"
    echo "GPA variants mode: ${GPA_VARIANTS_MODE}"
    echo "Skip eval: ${SKIP_EVAL}"
    echo "Skip existing: ${SKIP_EXISTING}"
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
    if [[ "${SKIP_EVAL}" == "1" ]]; then
        target_args+=(--skip-eval)
    fi
}

maybe_run() {
    local label="$1"
    local results_path="$2"
    shift 2

    if [[ "${SKIP_EXISTING}" == "1" && -f "${results_path}" ]]; then
        echo
        echo ">>> Skipping ${label}; found existing results at ${results_path}"
        return
    fi

    run_python "$@"
}

run_task_arithmetic() {
    local results_path="${RESULTS_ROOT}/task_arithmetic_results.json"
    local output_dir="${MERGED_ROOT}/task_arithmetic"
    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_task_arithmetic.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --lambdas "${DELTA_LAMBDAS}"
        --output-dir "${output_dir}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    maybe_run "Task Arithmetic sweep" "${results_path}" "${cmd[@]}"
}

run_ties() {
    local results_path="${RESULTS_ROOT}/ties_results.json"
    local output_dir="${MERGED_ROOT}/ties"
    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_ties.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "${TRIM_PERCENTAGES}"
        --lambdas "${DELTA_LAMBDAS}"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --output-dir "${output_dir}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    maybe_run "TIES sweep" "${results_path}" "${cmd[@]}"
}

run_dare_ties() {
    local results_path="${RESULTS_ROOT}/dare_ties_results.json"
    local output_dir="${MERGED_ROOT}/dare"
    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_dare.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --merge-methods ties
        --drop-probabilities "${DARE_DROP_PROBABILITIES}"
        --trim-percentages "${TRIM_PERCENTAGES}"
        --lambdas "${DELTA_LAMBDAS}"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --seed "${DARE_SEED}"
        --output-dir "${output_dir}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    maybe_run "DARE + TIES sweep" "${results_path}" "${cmd[@]}"
}

run_lr_knots() {
    local results_path="${RESULTS_ROOT}/lr_knots_results.json"
    local output_dir="${MERGED_ROOT}/lr_knots"
    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_lr_knots.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "${TRIM_PERCENTAGES}"
        --lambdas "${FACTOR_LAMBDAS}"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --output-dir "${output_dir}"
        --results-path "${results_path}"
    )
    append_eval_args cmd
    maybe_run "LR-KnOTS + TIES sweep" "${results_path}" "${cmd[@]}"
}

run_gpa_variant() {
    local slug="$1"
    shift

    local results_path="${RESULTS_ROOT}/gpa_ties_${slug}_results.json"
    local output_dir="${MERGED_ROOT}/gpa_ties/${slug}"
    local -a cmd=(
        "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
        --adapters-dir "${ADAPTERS_DIR}"
        --tasks "${TASK_ARRAY[@]}"
        --trim-percentages "${TRIM_PERCENTAGES}"
        --lambdas "${FACTOR_LAMBDAS}"
        --majority-sign-method "${MAJORITY_SIGN_METHOD}"
        --output-dir "${output_dir}"
        --results-path "${results_path}"
        "$@"
    )
    append_eval_args cmd
    maybe_run "GPA variant ${slug}" "${results_path}" "${cmd[@]}"
}

write_manifest() {
    export PROJECT_ROOT RESULTS_ROOT MERGED_ROOT ADAPTERS_DIR TASKS
    export DELTA_LAMBDAS FACTOR_LAMBDAS TRIM_PERCENTAGES DARE_DROP_PROBABILITIES
    export DARE_SEED MAJORITY_SIGN_METHOD EVAL_BATCH_SIZE MAX_EVAL_SAMPLES
    export SKIP_EVAL SKIP_EXISTING GPA_VARIANTS_MODE

    "${PYTHON_CMD}" - <<'PY'
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])

variant_files = ["gpa_ties_baseline_results.json"]
mode = os.environ["GPA_VARIANTS_MODE"]
if mode == "full":
    variant_files.extend(
        [
            "gpa_ties_dgpa_ties_results.json",
            "gpa_ties_dgpa_saties_results.json",
            "gpa_ties_dgpa_saties_wb_0p5_results.json",
            "gpa_ties_dgpa_saties_wb_1p0_results.json",
        ]
    )
elif mode == "minimal":
    variant_files.append("gpa_ties_dgpa_saties_wb_1p0_results.json")
elif mode == "none":
    pass
else:
    raise ValueError(f"Unsupported GPA_VARIANTS_MODE: {mode}")

payload = {
    "step": "week3_step3_1_hyperparameter_sweep",
    "tasks": os.environ["TASKS"].split(),
    "adapters_dir": os.environ["ADAPTERS_DIR"],
    "results_root": os.environ["RESULTS_ROOT"],
    "merged_root": os.environ["MERGED_ROOT"],
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
        "skip_eval": os.environ["SKIP_EVAL"] == "1",
        "skip_existing": os.environ["SKIP_EXISTING"] == "1",
        "gpa_variants_mode": mode,
    },
    "expected_results": {
        "task_arithmetic": "task_arithmetic_results.json",
        "ties": "ties_results.json",
        "dare_ties": "dare_ties_results.json",
        "lr_knots": "lr_knots_results.json",
        "gpa_ties_variants": variant_files,
    },
}

payload["existing_results"] = sorted(path.name for path in results_root.glob("*.json"))
(results_root / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"Saved Week 3 sweep manifest to {results_root / 'run_manifest.json'}")
PY
}

print_banner

if [[ "${WARMUP_EVAL_CACHE}" == "1" && "${SKIP_EVAL}" != "1" ]]; then
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

run_task_arithmetic
run_ties
run_dare_ties
run_lr_knots
run_gpa_variant baseline

case "${GPA_VARIANTS_MODE}" in
    full)
        run_gpa_variant dgpa_ties --normalise-a-factors
        run_gpa_variant dgpa_saties --normalise-a-factors --scale-aware-ties
        run_gpa_variant dgpa_saties_wb_0p5 --normalise-a-factors --scale-aware-ties --b-weight-alpha 0.5
        run_gpa_variant dgpa_saties_wb_1p0 --normalise-a-factors --scale-aware-ties --b-weight-alpha 1.0
        ;;
    minimal)
        run_gpa_variant dgpa_saties_wb_1p0 --normalise-a-factors --scale-aware-ties --b-weight-alpha 1.0
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
echo "Week 3 sweep complete."
echo "Results: ${RESULTS_ROOT}"
echo "Merged adapters: ${MERGED_ROOT}"
