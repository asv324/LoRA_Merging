#!/bin/bash
# Train all 5 QLoRA adapters sequentially on the single RTX 6000 Ada.
set -eu
set -o pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/adapters"

echo "Starting sequential training at ${TIMESTAMP}"

run_task() {
    local task="$1"
    local epochs="$2"
    local lr="$3"
    local batch="$4"
    local grad_accum="$5"
    local max_length="$6"
    local eval_steps="$7"
    local output_dir="${PROJECT_ROOT}/adapters/${task}"
    local log_path="${PROJECT_ROOT}/logs/${task}_${TIMESTAMP}.log"

    if [[ -f "${output_dir}/adapter_model.safetensors" && -f "${output_dir}/eval_metrics.json" ]]; then
        echo "Skipping ${task}: existing adapter and eval metrics found."
        return 0
    fi

    echo "Training ${task} -> ${output_dir}"
    python "${PROJECT_ROOT}/scripts/train.py" --task "${task}" \
        --output_dir "${output_dir}" \
        --epochs "${epochs}" --lr "${lr}" --per_device_batch "${batch}" --grad_accum "${grad_accum}" \
        --max_length "${max_length}" --eval_steps "${eval_steps}" \
        2>&1 | tee "${log_path}"
}

run_task sst2 3 2e-4 32 1 128 500
run_task mnli 2 2e-4 32 1 256 1000
run_task qnli 3 2e-4 32 1 256 500
run_task cola 5 1e-4 32 1 128 100
run_task rte 10 1e-4 32 1 256 50

echo "All training runs complete."
