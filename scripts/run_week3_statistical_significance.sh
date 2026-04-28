#!/bin/bash
# Run Week 3 Step 3.3 statistical-significance reruns across multiple seeds.
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

SEEDS="${SEEDS:-42 43 44}"
read -r -a SEED_ARRAY <<< "${SEEDS}"

METHODS="${METHODS:-gpa_baseline ties lr_knots gpa_dgpa_saties_wb_0p5}"
read -r -a METHOD_ARRAY <<< "${METHODS}"

# Optional per-method lambda overrides, space-separated "alias=value" pairs.
# Default is empty so every method uses the hyperparameters from BEST_HPARAMS_PATH.
# After the 2026-04-21 restored-head HP re-sweep, ``gpa_dgpa_saties_wb_0p5`` is
# argmaxed at lambda=1.0 (trim=30, …); pinning 0.25 here was a legacy Week-3
# random-head headline and silently diverged from ``results/best_hparams.json``.
# To reproduce an old paper draft, set explicitly, e.g.:
#   LAMBDA_OVERRIDES="gpa_dgpa_saties_wb_0p5=0.25"
LAMBDA_OVERRIDES="${LAMBDA_OVERRIDES:-}"

# Space-separated list of aliases that should be used as comparison baselines in
# the aggregated summary. The proposal's success criterion is stated against raw
# TIES, so we always emit "<alias>_vs_ties" deltas in addition to the historical
# "<alias>_vs_gpa_baseline" ones. Any baseline listed here must also appear in
# METHODS so that its per-seed results are available when the summary is built.
COMPARISON_BASELINES="${COMPARISON_BASELINES:-gpa_baseline ties}"

BEST_HPARAMS_PATH="${BEST_HPARAMS_PATH:-${PROJECT_ROOT}/results/best_hparams.json}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results/statistical_significance_restored_heads}"
MERGED_ROOT="${MERGED_ROOT:-${PROJECT_ROOT}/merged_adapters/statistical_significance_restored_heads_tmp}"
# NOTE: do not fold this into a `${VAR:-default}` expansion. Bash parameter
# expansion does not treat bare `{...}` as balanced inside the default word,
# so a trailing `}` in the default ("seed_{seed}}") would be consumed as the
# outer closer on the unset branch and re-appended as a literal byte on the
# set branch, silently corrupting any user-supplied ADAPTERS_SEED_TEMPLATE
# with a trailing "}". Use an explicit `if` guard instead.
if [[ -z "${ADAPTERS_SEED_TEMPLATE:-}" ]]; then
    ADAPTERS_SEED_TEMPLATE="${PROJECT_ROOT}/adapters/seed_{seed}"
fi
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/statistical_significance}"
SEED_VARIANCE_PATH="${SEED_VARIANCE_PATH:-${PROJECT_ROOT}/results/seed_variance_restored_heads.json}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B}"
EVAL_CACHE_ROOT="${EVAL_CACHE_ROOT:-${PROJECT_ROOT}/.cache/hf_eval}"
WARMUP_EVAL_CACHE="${WARMUP_EVAL_CACHE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

MAJORITY_SIGN_METHOD="${MAJORITY_SIGN_METHOD:-total}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DELETE_AFTER_EVAL="${DELETE_AFTER_EVAL:-1}"

TRAIN_MISSING_ADAPTERS="${TRAIN_MISSING_ADAPTERS:-0}"
TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-none}"

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

if [[ "${ADAPTERS_SEED_TEMPLATE}" != *"{seed}"* && "${#SEED_ARRAY[@]}" -gt 1 ]]; then
    echo "ADAPTERS_SEED_TEMPLATE must include '{seed}' when running multiple seeds." >&2
    exit 1
fi

print_banner() {
    echo
    echo "Week 3 Step 3.3 statistical significance"
    echo "Python: ${PYTHON_CMD}"
    echo "Tasks: ${TASKS}"
    echo "Seeds: ${SEEDS}"
    echo "Methods: ${METHODS}"
    echo "Best hparams path: ${BEST_HPARAMS_PATH}"
    echo "Adapters seed template: ${ADAPTERS_SEED_TEMPLATE}"
    echo "Results root: ${RESULTS_ROOT}"
    echo "Temporary merged root: ${MERGED_ROOT}"
    echo "Evaluation cache root: ${EVAL_CACHE_ROOT}"
    echo "Warmup eval cache: ${WARMUP_EVAL_CACHE}"
    echo "Local files only: ${LOCAL_FILES_ONLY}"
    echo "Train missing adapters: ${TRAIN_MISSING_ADAPTERS}"
    echo "Skip existing: ${SKIP_EXISTING}"
    echo "Delete merged adapters after eval: ${DELETE_AFTER_EVAL}"
    echo "Lambda overrides: ${LAMBDA_OVERRIDES:-<none>}"
    echo "Comparison baselines: ${COMPARISON_BASELINES}"
    echo "Seed-variance summary: ${SEED_VARIANCE_PATH}"
}

lookup_lambda_override() {
    local method_alias="$1"
    local pair
    for pair in ${LAMBDA_OVERRIDES}; do
        local key="${pair%%=*}"
        local value="${pair#*=}"
        if [[ "${key}" == "${method_alias}" ]]; then
            printf '%s' "${value}"
            return 0
        fi
    done
    printf ''
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

resolve_adapters_dir() {
    local seed="$1"
    printf '%s\n' "${ADAPTERS_SEED_TEMPLATE//\{seed\}/${seed}}"
}

adapter_ready() {
    local task_dir="$1"
    [[ -f "${task_dir}/adapter_model.safetensors" && -f "${task_dir}/eval_metrics.json" ]]
}

append_train_args() {
    local task="$1"
    local -n target_args=$2
    case "${task}" in
        sst2)
            target_args+=(--epochs 3 --lr 2e-4 --per_device_batch 32 --grad_accum 1 --max_length 128 --eval_steps 500)
            ;;
        mnli)
            target_args+=(--epochs 2 --lr 2e-4 --per_device_batch 32 --grad_accum 1 --max_length 256 --eval_steps 1000)
            ;;
        qnli)
            target_args+=(--epochs 3 --lr 2e-4 --per_device_batch 32 --grad_accum 1 --max_length 256 --eval_steps 500)
            ;;
        cola)
            target_args+=(--epochs 5 --lr 1e-4 --per_device_batch 32 --grad_accum 1 --max_length 128 --eval_steps 100)
            ;;
        rte)
            target_args+=(--epochs 10 --lr 1e-4 --per_device_batch 32 --grad_accum 1 --max_length 256 --eval_steps 50)
            ;;
        *)
            echo "Unsupported training task: ${task}" >&2
            exit 1
            ;;
    esac
}

train_task_for_seed() {
    local seed="$1"
    local task="$2"
    local adapters_dir="$3"
    local output_dir="${adapters_dir}/${task}"
    local log_path="${LOG_ROOT}/seed_${seed}_${task}.log"
    mkdir -p "${output_dir}" "${LOG_ROOT}"

    local -a cmd=(
        "${PROJECT_ROOT}/scripts/train.py"
        --task "${task}"
        --model-name "${MODEL_NAME}"
        --output-dir "${output_dir}"
        --seed "${seed}"
        --report-to "${TRAIN_REPORT_TO}"
    )
    append_train_args "${task}" cmd

    echo
    echo ">>> Training missing adapter for seed ${seed}, task ${task}"
    printf '>>> '
    printf '%q ' "${PYTHON_CMD}" "${cmd[@]}"
    echo
    "${PYTHON_CMD}" "${cmd[@]}" 2>&1 | tee "${log_path}"
}

ensure_seed_adapters() {
    local seed="$1"
    local adapters_dir
    adapters_dir="$(resolve_adapters_dir "${seed}")"
    mkdir -p "${adapters_dir}"

    local missing=0
    local task
    for task in "${TASK_ARRAY[@]}"; do
        if ! adapter_ready "${adapters_dir}/${task}"; then
            missing=1
            if [[ "${TRAIN_MISSING_ADAPTERS}" == "1" ]]; then
                train_task_for_seed "${seed}" "${task}" "${adapters_dir}"
            fi
        fi
    done

    if [[ "${missing}" == "1" && "${TRAIN_MISSING_ADAPTERS}" != "1" ]]; then
        echo "Missing seed-specific adapters under ${adapters_dir}." >&2
        echo "Set TRAIN_MISSING_ADAPTERS=1 to train them automatically." >&2
        exit 1
    fi
}

get_method_config() {
    local method_alias="$1"
    # Fields are joined with the ASCII Unit Separator (\x1F) instead of \t.
    # Tab is treated as IFS whitespace by bash, which collapses sequences of
    # tabs into a single delimiter and silently shifts every later field when a
    # middle field is empty (e.g. drop_probability=null). Using a non-whitespace
    # delimiter forces `read` to preserve empty fields and keep alignment.
    "${PYTHON_CMD}" - "${BEST_HPARAMS_PATH}" "${method_alias}" <<'PY'
import json
import sys
from pathlib import Path

best_hparams_path = Path(sys.argv[1])
method_alias = sys.argv[2]

payload = json.loads(best_hparams_path.read_text(encoding="utf-8"))
methods = payload.get("methods", {})

record = methods.get(method_alias)
if record is None:
    record = methods.get("gpa_variants", {}).get(method_alias)
if record is None:
    raise SystemExit(f"Method alias '{method_alias}' not found in {best_hparams_path}")

hp = record.get("hyperparameters", {})
fields = [
    str(record.get("method_key", method_alias)),
    str(record.get("display_name", method_alias)),
    "" if hp.get("lambda") is None else str(hp.get("lambda")),
    "" if hp.get("trim_percentage") is None else str(hp.get("trim_percentage")),
    "" if hp.get("drop_probability") is None else str(hp.get("drop_probability")),
    "" if hp.get("b_weight_alpha") is None else str(hp.get("b_weight_alpha")),
    "1" if hp.get("normalise_a_factors") else "0",
    "1" if hp.get("scale_aware_ties") else "0",
    str(record.get("source_path", "")),
]
print("\x1f".join(fields))
PY
}

run_config() {
    local label="$1"
    local results_path="$2"
    local cleanup_target="$3"
    shift 3

    if [[ "${SKIP_EXISTING}" == "1" && -f "${results_path}" ]]; then
        echo
        echo ">>> Skipping ${label}; found existing results at ${results_path}"
        return
    fi

    mkdir -p "$(dirname "${results_path}")"
    run_python "$@"
    if [[ ! -f "${results_path}" ]]; then
        echo "Expected results file was not created: ${results_path}" >&2
        exit 1
    fi
    cleanup_dir "${cleanup_target}"
}

run_method_for_seed() {
    local seed="$1"
    local method_alias="$2"

    local adapters_dir
    adapters_dir="$(resolve_adapters_dir "${seed}")"

    local config_line
    config_line="$(get_method_config "${method_alias}")"

    local actual_key display_name lambda_value trim_percentage drop_probability b_weight_alpha
    local normalise_a_factors scale_aware_ties source_path
    # Use ASCII Unit Separator (\x1F) so empty middle fields (e.g. a null
    # drop_probability) are preserved as empty strings instead of collapsing
    # consecutive delimiters and shifting every subsequent column.
    IFS=$'\x1f' read -r actual_key display_name lambda_value trim_percentage drop_probability b_weight_alpha \
        normalise_a_factors scale_aware_ties source_path <<< "${config_line}"

    local lambda_override
    lambda_override="$(lookup_lambda_override "${method_alias}")"
    if [[ -n "${lambda_override}" ]]; then
        if [[ -n "${lambda_value}" && "${lambda_value}" != "${lambda_override}" ]]; then
            echo ">>> Applying LAMBDA_OVERRIDES for ${method_alias}: ${lambda_value} -> ${lambda_override}"
        else
            echo ">>> Pinning lambda for ${method_alias} to ${lambda_override} via LAMBDA_OVERRIDES"
        fi
        lambda_value="${lambda_override}"
    fi

    if [[ -z "${lambda_value}" ]]; then
        echo "Method ${method_alias} does not define a lambda value in ${BEST_HPARAMS_PATH}." >&2
        exit 1
    fi

    local results_path="${RESULTS_ROOT}/seed_${seed}/${method_alias}.json"
    local output_dir="${MERGED_ROOT}/seed_${seed}/${method_alias}"

    if [[ -f "${results_path}" ]]; then
        local stale_state
        stale_state="$(
            EXPECTED_LAMBDA="${lambda_value}" \
            EXPECTED_TRIM="${trim_percentage}" \
            EXPECTED_DROP="${drop_probability}" \
            EXPECTED_BWA="${b_weight_alpha}" \
            EXPECTED_NORM_A="${normalise_a_factors}" \
            EXPECTED_SATIES="${scale_aware_ties}" \
            EXPECTED_KEY="${actual_key}" \
            "${PYTHON_CMD}" - "${results_path}" <<'PY'
import json, math, os, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def parse_bool(value):
    if value is None or value == "":
        return None
    return value == "1"

def floats_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)

expected = {
    "lambda": parse_float(os.environ.get("EXPECTED_LAMBDA")),
    "trim_percentage": parse_float(os.environ.get("EXPECTED_TRIM")),
    "drop_probability": parse_float(os.environ.get("EXPECTED_DROP")),
    "b_weight_alpha": parse_float(os.environ.get("EXPECTED_BWA")),
    "normalise_a_factors": parse_bool(os.environ.get("EXPECTED_NORM_A")),
    "scale_aware_ties": parse_bool(os.environ.get("EXPECTED_SATIES")),
}
expected_key = os.environ.get("EXPECTED_KEY", "")

runs = payload.get("runs") or []
if not runs:
    print("STALE:no_runs_recorded")
    sys.exit(0)
run = runs[0]

mismatches = []

# Method-key sanity: GPA-family JSONs explicitly tag the variant they ran.
variant_label = payload.get("variant_label") or run.get("variant_label")
if expected_key.startswith("gpa_") and variant_label is not None:
    if expected_key == "gpa_baseline" and variant_label != "GPA+TIES":
        mismatches.append(f"variant_label={variant_label}!=GPA+TIES")
    if expected_key == "gpa_dgpa_ties" and variant_label != "dGPA+TIES":
        mismatches.append(f"variant_label={variant_label}!=dGPA+TIES")
    if expected_key == "gpa_dgpa_saties" and variant_label != "dGPA+saTIES":
        mismatches.append(f"variant_label={variant_label}!=dGPA+saTIES")
    if expected_key == "gpa_dgpa_saties_wb_0p5" and variant_label != "dGPA+saTIES+wB(0.5)":
        mismatches.append(f"variant_label={variant_label}!=dGPA+saTIES+wB(0.5)")
    if expected_key == "gpa_dgpa_saties_wb_1p0" and variant_label not in {"dGPA+saTIES+wB(1)", "dGPA+saTIES+wB(1.0)"}:
        mismatches.append(f"variant_label={variant_label}!=dGPA+saTIES+wB(1.0)")

# Numeric / boolean hyperparameters that the merge scripts surface in run[0].
for field, parser in (
    ("lambda", parse_float),
    ("trim_percentage", parse_float),
    ("drop_probability", parse_float),
    ("b_weight_alpha", parse_float),
):
    target = expected[field]
    if target is None:
        continue
    actual = run.get(field)
    if actual is None:
        actual = payload.get(field)
    actual = parser(actual) if not isinstance(actual, (int, float)) else float(actual)
    if not floats_equal(actual, target):
        mismatches.append(f"{field}={actual}!={target}")

for field in ("normalise_a_factors", "scale_aware_ties"):
    target = expected[field]
    if target is None:
        continue
    actual = run.get(field)
    if actual is None:
        actual = payload.get(field)
        if actual is None:
            gpa_block = payload.get("gpa") or {}
            actual = gpa_block.get(field)
    if actual is None or bool(actual) != target:
        mismatches.append(f"{field}={actual}!={target}")

if mismatches:
    print("STALE:" + "|".join(mismatches))
else:
    print("MATCH")
PY
        )"
        if [[ "${stale_state}" == STALE:* ]]; then
            local stale_reason="${stale_state#STALE:}"
            echo ">>> Stale results for ${method_alias} seed=${seed} (${stale_reason}); removing ${results_path}"
            rm -f "${results_path}"
        fi
    fi

    case "${actual_key}" in
        task_arithmetic)
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_task_arithmetic.py"
                --adapters-dir "${adapters_dir}"
                --tasks "${TASK_ARRAY[@]}"
                --lambdas "${lambda_value}"
                --output-dir "${output_dir}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "${display_name} seed=${seed}" "${results_path}" "${output_dir}" "${cmd[@]}"
            ;;
        ties)
            if [[ -z "${trim_percentage}" ]]; then
                echo "Method ${method_alias} requires trim_percentage in ${BEST_HPARAMS_PATH}." >&2
                exit 1
            fi
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_ties.py"
                --adapters-dir "${adapters_dir}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim_percentage}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${output_dir}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "${display_name} seed=${seed}" "${results_path}" "${output_dir}" "${cmd[@]}"
            ;;
        dare_ties)
            if [[ -z "${trim_percentage}" || -z "${drop_probability}" ]]; then
                echo "Method ${method_alias} requires trim_percentage and drop_probability in ${BEST_HPARAMS_PATH}." >&2
                exit 1
            fi
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_dare.py"
                --adapters-dir "${adapters_dir}"
                --tasks "${TASK_ARRAY[@]}"
                --merge-methods ties
                --drop-probabilities "${drop_probability}"
                --trim-percentages "${trim_percentage}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --seed "${seed}"
                --output-dir "${output_dir}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "${display_name} seed=${seed}" "${results_path}" "${output_dir}" "${cmd[@]}"
            ;;
        lr_knots)
            if [[ -z "${trim_percentage}" ]]; then
                echo "Method ${method_alias} requires trim_percentage in ${BEST_HPARAMS_PATH}." >&2
                exit 1
            fi
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_lr_knots.py"
                --adapters-dir "${adapters_dir}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim_percentage}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${output_dir}"
                --results-path "${results_path}"
            )
            append_eval_args cmd
            run_config "${display_name} seed=${seed}" "${results_path}" "${output_dir}" "${cmd[@]}"
            ;;
        gpa_baseline|gpa_dgpa_ties|gpa_dgpa_saties|gpa_dgpa_saties_wb_0p5|gpa_dgpa_saties_wb_1p0)
            if [[ -z "${trim_percentage}" ]]; then
                echo "Method ${method_alias} requires trim_percentage in ${BEST_HPARAMS_PATH}." >&2
                exit 1
            fi
            local -a cmd=(
                "${PROJECT_ROOT}/scripts/merge_gpa_ties.py"
                --adapters-dir "${adapters_dir}"
                --tasks "${TASK_ARRAY[@]}"
                --trim-percentages "${trim_percentage}"
                --lambdas "${lambda_value}"
                --majority-sign-method "${MAJORITY_SIGN_METHOD}"
                --output-dir "${output_dir}"
                --results-path "${results_path}"
            )
            if [[ "${normalise_a_factors}" == "1" ]]; then
                cmd+=(--normalise-a-factors)
            fi
            if [[ "${scale_aware_ties}" == "1" ]]; then
                cmd+=(--scale-aware-ties)
            fi
            if [[ -n "${b_weight_alpha}" && "${b_weight_alpha}" != "0" && "${b_weight_alpha}" != "0.0" ]]; then
                cmd+=(--b-weight-alpha "${b_weight_alpha}")
            fi
            append_eval_args cmd
            run_config "${display_name} seed=${seed}" "${results_path}" "${output_dir}" "${cmd[@]}"
            ;;
        *)
            echo "Unsupported method key '${actual_key}' for alias '${method_alias}'." >&2
            exit 1
            ;;
    esac
}

write_summary() {
    export PROJECT_ROOT RESULTS_ROOT BEST_HPARAMS_PATH TASKS SEEDS METHODS ADAPTERS_SEED_TEMPLATE
    export TRAIN_MISSING_ADAPTERS MAJORITY_SIGN_METHOD EVAL_BATCH_SIZE MAX_EVAL_SAMPLES LAMBDA_OVERRIDES
    export SEED_VARIANCE_PATH COMPARISON_BASELINES

    "${PYTHON_CMD}" - <<'PY'
import json
import statistics
import os
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data import get_task_config

results_root = Path(os.environ["RESULTS_ROOT"])
best_hparams_path = Path(os.environ["BEST_HPARAMS_PATH"])
tasks = os.environ["TASKS"].split()
seeds = [int(piece) for piece in os.environ["SEEDS"].split()]
method_aliases = os.environ["METHODS"].split()

best_hparams = json.loads(best_hparams_path.read_text(encoding="utf-8"))
methods_payload = best_hparams.get("methods", {})

lambda_overrides: dict[str, float] = {}
for pair in os.environ.get("LAMBDA_OVERRIDES", "").split():
    if "=" not in pair:
        continue
    key, _, value = pair.partition("=")
    key = key.strip()
    value = value.strip()
    if key and value:
        lambda_overrides[key] = float(value)

def resolve_record(alias: str) -> dict:
    record = methods_payload.get(alias)
    if record is None:
        record = methods_payload.get("gpa_variants", {}).get(alias)
    if record is None:
        raise KeyError(f"Method alias '{alias}' not found in {best_hparams_path}")
    return record

def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot compute mean/std for empty values")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"values": values, "mean": mean, "std": std}

def extract_primary_metric(task: str, task_metrics: dict) -> tuple[str, float]:
    metric_name = get_task_config(task).metric_for_best_model
    for candidate in (f"eval_{metric_name}", metric_name):
        if candidate in task_metrics:
            return metric_name, float(task_metrics[candidate])
    raise KeyError(f"Primary metric '{metric_name}' not found for task '{task}'.")

summary = {
    "step": "week3_step3_3_statistical_significance",
    "best_hparams_path": str(best_hparams_path),
    "tasks": tasks,
    "seeds": seeds,
    "method_aliases": method_aliases,
    "adapters_seed_template": os.environ["ADAPTERS_SEED_TEMPLATE"],
    "runtime": {
        "train_missing_adapters": os.environ["TRAIN_MISSING_ADAPTERS"] == "1",
        "majority_sign_method": os.environ["MAJORITY_SIGN_METHOD"],
        "eval_batch_size": int(os.environ["EVAL_BATCH_SIZE"]),
        "max_eval_samples": os.environ["MAX_EVAL_SAMPLES"] or None,
        "lambda_overrides": lambda_overrides,
    },
    "methods": {},
    "comparisons": {},
}

for alias in method_aliases:
    record = resolve_record(alias)
    seed_entries = []
    average_values = []
    per_task_values = {task: [] for task in tasks}
    per_task_metric_names = {}

    for seed in seeds:
        result_path = results_root / f"seed_{seed}" / f"{alias}.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        run = payload["runs"][0]
        evaluation = run.get("evaluation")
        if not isinstance(evaluation, dict):
            raise KeyError(
                f"Result file {result_path} does not contain per-task 'evaluation' metrics; "
                "rerun the merge script without --skip-eval."
            )

        task_block = {}
        primary_values = []
        for task in tasks:
            task_metrics = evaluation.get(task)
            if not isinstance(task_metrics, dict):
                raise KeyError(f"Task '{task}' missing from evaluation block in {result_path}.")
            metric_name, metric_value = extract_primary_metric(task, task_metrics)
            per_task_metric_names[task] = metric_name
            per_task_values[task].append(metric_value)
            task_block[task] = {
                "metric": metric_name,
                "value": metric_value,
            }
            primary_values.append(metric_value)

        average_primary_score = sum(primary_values) / len(primary_values)
        average_values.append(average_primary_score)

        seed_entries.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "average_primary_score": average_primary_score,
                "primary_metrics": task_block,
            }
        )

    hyperparameters = dict(record.get("hyperparameters", {}))
    if alias in lambda_overrides:
        hyperparameters["lambda"] = lambda_overrides[alias]
        hyperparameters["lambda_override_applied"] = True

    summary["methods"][alias] = {
        "display_name": record["display_name"],
        "selected_method_key": record["method_key"],
        "selected_source_path": record.get("source_path"),
        "hyperparameters": hyperparameters,
        "per_seed": seed_entries,
        "average_primary_score": mean_std(average_values),
        "tasks": {
            task: {
                "metric": per_task_metric_names[task],
                **mean_std(per_task_values[task]),
            }
            for task in tasks
        },
    }

comparison_baselines = [
    alias
    for alias in os.environ.get("COMPARISON_BASELINES", "gpa_baseline").split()
    if alias
]
for baseline_alias in comparison_baselines:
    if baseline_alias not in summary["methods"]:
        # A configured baseline is allowed to be missing from METHODS (e.g. a
        # partial rerun that only regenerates one method); skip silently rather
        # than aborting so the rest of the summary still lands on disk.
        continue
    baseline = summary["methods"][baseline_alias]
    baseline_seed_lookup = {entry["seed"]: entry for entry in baseline["per_seed"]}
    for alias in method_aliases:
        if alias == baseline_alias:
            continue
        other = summary["methods"][alias]
        other_seed_lookup = {entry["seed"]: entry for entry in other["per_seed"]}
        delta_average = []
        delta_by_task = {task: [] for task in tasks}
        for seed in seeds:
            delta_average.append(
                float(other_seed_lookup[seed]["average_primary_score"])
                - float(baseline_seed_lookup[seed]["average_primary_score"])
            )
            for task in tasks:
                delta_by_task[task].append(
                    float(other_seed_lookup[seed]["primary_metrics"][task]["value"])
                    - float(baseline_seed_lookup[seed]["primary_metrics"][task]["value"])
                )

        task_summaries = {
            task: {
                "metric": baseline["tasks"][task]["metric"],
                **mean_std(values),
            }
            for task, values in delta_by_task.items()
        }
        positive_task_count = sum(1 for task in tasks if task_summaries[task]["mean"] > 0.0)
        average_delta_summary = mean_std(delta_average)

        summary["comparisons"][f"{alias}_vs_{baseline_alias}"] = {
            "baseline_alias": baseline_alias,
            "comparison_alias": alias,
            "delta_direction": "comparison_minus_baseline",
            "average_primary_score_delta": average_delta_summary,
            "task_deltas": task_summaries,
            "positive_mean_delta_task_count": positive_task_count,
            "average_primary_score_delta_ge_0p01": average_delta_summary["mean"] >= 0.01,
            "proposal_success_criterion_met": (
                average_delta_summary["mean"] >= 0.01 and positive_task_count >= 3
            ),
        }

output_path = results_root / "summary.json"
output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved statistical-significance summary to {output_path}")

seed_variance_path = Path(os.environ["SEED_VARIANCE_PATH"])
seed_variance_path.parent.mkdir(parents=True, exist_ok=True)
seed_variance = {
    "step": "week3_step3_3_seed_variance",
    "source_summary": str(output_path),
    "tasks": tasks,
    "seeds": seeds,
    "method_aliases": method_aliases,
    "lambda_overrides": lambda_overrides,
    "methods": {
        alias: {
            "display_name": block["display_name"],
            "selected_method_key": block["selected_method_key"],
            "hyperparameters": block["hyperparameters"],
            "average_primary_score": {
                "mean": block["average_primary_score"]["mean"],
                "std": block["average_primary_score"]["std"],
                "values": block["average_primary_score"]["values"],
            },
            "tasks": {
                task: {
                    "metric": payload["metric"],
                    "mean": payload["mean"],
                    "std": payload["std"],
                    "values": payload["values"],
                }
                for task, payload in block["tasks"].items()
            },
        }
        for alias, block in summary["methods"].items()
    },
    "comparisons": summary["comparisons"],
}
seed_variance_path.write_text(json.dumps(seed_variance, indent=2), encoding="utf-8")
print(f"Saved seed-variance artifact to {seed_variance_path}")
PY
}

write_manifest() {
    export RESULTS_ROOT MERGED_ROOT BEST_HPARAMS_PATH TASKS SEEDS METHODS ADAPTERS_SEED_TEMPLATE
    export TRAIN_MISSING_ADAPTERS MAJORITY_SIGN_METHOD EVAL_BATCH_SIZE MAX_EVAL_SAMPLES
    export SKIP_EXISTING DELETE_AFTER_EVAL MODEL_NAME EVAL_CACHE_ROOT WARMUP_EVAL_CACHE LOCAL_FILES_ONLY
    export LAMBDA_OVERRIDES COMPARISON_BASELINES

    "${PYTHON_CMD}" - <<'PY'
import json
import os
from pathlib import Path

results_root = Path(os.environ["RESULTS_ROOT"])
result_files = sorted(str(path.relative_to(results_root)) for path in results_root.rglob("*.json"))

manifest = {
    "step": "week3_step3_3_statistical_significance_low_storage",
    "best_hparams_path": os.environ["BEST_HPARAMS_PATH"],
    "tasks": os.environ["TASKS"].split(),
    "seeds": [int(piece) for piece in os.environ["SEEDS"].split()],
    "method_aliases": os.environ["METHODS"].split(),
    "adapters_seed_template": os.environ["ADAPTERS_SEED_TEMPLATE"],
    "results_root": os.environ["RESULTS_ROOT"],
    "temporary_merged_root": os.environ["MERGED_ROOT"],
    "evaluation_cache_root": os.environ["EVAL_CACHE_ROOT"],
    "runtime": {
        "train_missing_adapters": os.environ["TRAIN_MISSING_ADAPTERS"] == "1",
        "majority_sign_method": os.environ["MAJORITY_SIGN_METHOD"],
        "eval_batch_size": int(os.environ["EVAL_BATCH_SIZE"]),
        "max_eval_samples": os.environ["MAX_EVAL_SAMPLES"] or None,
        "skip_existing": os.environ["SKIP_EXISTING"] == "1",
        "delete_after_eval": os.environ["DELETE_AFTER_EVAL"] == "1",
        "model_name": os.environ["MODEL_NAME"],
        "warmup_eval_cache": os.environ["WARMUP_EVAL_CACHE"] == "1",
        "local_files_only": os.environ["LOCAL_FILES_ONLY"] == "1",
        "lambda_overrides": os.environ.get("LAMBDA_OVERRIDES", ""),
        "comparison_baselines": [
            alias for alias in os.environ.get("COMPARISON_BASELINES", "").split() if alias
        ],
    },
    "result_file_count": len(result_files),
    "result_files": result_files,
}

(results_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Saved statistical-significance manifest to {results_root / 'run_manifest.json'}")
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

for seed in "${SEED_ARRAY[@]}"; do
    ensure_seed_adapters "${seed}"
    for method_alias in "${METHOD_ARRAY[@]}"; do
        run_method_for_seed "${seed}" "${method_alias}"
    done
done

write_summary
write_manifest

cleanup_merged_root() {
    if [[ "${DELETE_AFTER_EVAL}" != "1" ]]; then
        return
    fi
    if [[ ! -d "${MERGED_ROOT}" ]]; then
        return
    fi
    # Remove all empty directories under MERGED_ROOT, then MERGED_ROOT itself if it
    # ends up empty. Any remaining non-empty directory is preserved so nothing
    # unexpected gets deleted.
    find "${MERGED_ROOT}" -mindepth 1 -depth -type d -empty -delete 2>/dev/null || true
    if [[ -d "${MERGED_ROOT}" ]] && [[ -z "$(ls -A "${MERGED_ROOT}" 2>/dev/null)" ]]; then
        rmdir "${MERGED_ROOT}" 2>/dev/null || true
    fi
}

cleanup_merged_root

echo
echo "Week 3 statistical-significance reruns complete."
echo "Results: ${RESULTS_ROOT}"
