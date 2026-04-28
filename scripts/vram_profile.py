"""Profile VRAM usage for the planned QLoRA training setup.

Implements Step B.2 from the revised implementation plan:
- Load quantized Qwen2.5-1.5B and measure base VRAM.
- Attach LoRA (r=16, attention + MLP projections).
- Run a single forward/backward pass at batch_size=32, seq_len=256.
- Save a structured report to `results/vram_profile.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "vram_profile.json"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / 1e9


def clear_cuda_state(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def get_memory_snapshot(device: torch.device) -> Dict[str, float]:
    return {
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved(device)),
        "peak_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gb": bytes_to_gb(torch.cuda.max_memory_reserved(device)),
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def build_dummy_batch(
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    num_labels: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
    attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long, device=device)
    labels = torch.randint(0, max(num_labels, 2), (batch_size,), device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_training_step(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> Dict[str, object]:
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    num_labels = int(getattr(model.config, "num_labels", 2))
    dummy_batch = build_dummy_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=max(vocab_size, 2),
        num_labels=num_labels,
        device=device,
    )

    clear_cuda_state(device)
    torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    model.train()

    started_at = time.perf_counter()
    outputs = model(**dummy_batch)
    loss = outputs.loss
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started_at

    snapshot = get_memory_snapshot(device)
    result: Dict[str, object] = {
        "status": "ok",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "loss": float(loss.detach().item()),
        "elapsed_seconds": float(elapsed_seconds),
        **snapshot,
    }

    model.zero_grad(set_to_none=True)
    return result


def try_training_step(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    target_limit_gb: float,
) -> Dict[str, object]:
    try:
        result = run_training_step(model, batch_size=batch_size, sequence_length=sequence_length, device=device)
        result["within_target"] = bool(result["peak_allocated_gb"] <= target_limit_gb)
        return result
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        clear_cuda_state(device)
        return {
            "status": "oom",
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "within_target": False,
            "error": str(exc),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--total-gpu-memory-gb", type=float, default=48.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def ensure_pad_token_id(model: torch.nn.Module) -> Dict[str, object]:
    config = model.config
    original_pad_token_id = getattr(config, "pad_token_id", None)
    eos_token_id = getattr(config, "eos_token_id", None)

    if original_pad_token_id is None:
        config.pad_token_id = eos_token_id if eos_token_id is not None else 0

    return {
        "pad_token_id": int(config.pad_token_id),
        "pad_token_source": "existing" if original_pad_token_id is not None else "fallback_from_eos_or_zero",
        "original_pad_token_id": None if original_pad_token_id is None else int(original_pad_token_id),
        "eos_token_id": None if eos_token_id is None else int(eos_token_id),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM profiling.")

    clear_cuda_state(device)
    torch.cuda.reset_peak_memory_stats(device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    load_started_at = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        num_labels=args.num_labels,
        device_map={"": device.index if device.index is not None else 0},
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    padding_info = ensure_pad_token_id(model)
    base_load_seconds = time.perf_counter() - load_started_at
    base_model_memory = get_memory_snapshot(device)

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    lora_memory = get_memory_snapshot(device)
    parameter_counts = count_parameters(model)

    reference_step = try_training_step(
        model=model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
        target_limit_gb=args.total_gpu_memory_gb,
    )
    if reference_step["status"] != "ok":
        raise RuntimeError(f"Reference profiling step failed: {reference_step['error']}")

    remaining_headroom_gb = max(args.total_gpu_memory_gb - float(reference_step["peak_allocated_gb"]), 0.0)
    expected_range = {
        "peak_vram_gb_min": 3.0,
        "peak_vram_gb_max": 5.0,
    }
    within_expected_range = expected_range["peak_vram_gb_min"] <= float(reference_step["peak_allocated_gb"]) <= expected_range["peak_vram_gb_max"]

    payload = {
        "experiment": "vram_profile",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "device": {
            "requested_device": args.device,
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "profiling_config": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "total_gpu_memory_gb": args.total_gpu_memory_gb,
            "padding": padding_info,
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": DEFAULT_TARGET_MODULES,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "SEQ_CLS",
            },
        },
        "base_model": {
            "load_seconds": float(base_load_seconds),
            **base_model_memory,
        },
        "lora_model": {
            **parameter_counts,
            **lora_memory,
        },
        "reference_training_step": reference_step,
        "summary": {
            "remaining_headroom_gb": remaining_headroom_gb,
            "within_expected_peak_range": within_expected_range,
            "expected_peak_range_gb": expected_range,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")
    print(f"Base model peak VRAM: {payload['base_model']['peak_allocated_gb']:.3f} GB")
    print(f"Reference LoRA training peak VRAM: {reference_step['peak_allocated_gb']:.3f} GB")
    print(f"Remaining headroom: {remaining_headroom_gb:.3f} GB")


if __name__ == "__main__":
    main()