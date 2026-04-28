"""Standalone evaluation for trained GLUE LoRA adapters."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Fixed seed for the classifier-head random init when the merged adapter does
# not restore `score.weight`. The source-task adapters are saved without their
# trained head in the merged state dict, so `AutoModelForSequenceClassification`
# re-initialises `score.weight` on every load. Without a fixed seed this draw
# varies across processes, which adds spurious noise on binary tasks (notably
# SST-2) when comparing merge variants that produce identical effective deltas.
# Evaluation remains deterministic relative to the merged adapter as long as we
# reseed right before the model is instantiated.
DEFAULT_EVAL_SEED = 0


def _resolve_eval_seed() -> int:
    raw = os.environ.get("LORA_MERGE_EVAL_SEED")
    if raw is None or raw.strip() == "":
        return DEFAULT_EVAL_SEED
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"LORA_MERGE_EVAL_SEED must be an integer, got {raw!r}"
        ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import (
    build_compute_metrics_fn,
    get_task_config,
    load_task_dataset,
    resolve_shared_eval_cache_root,
    tokenize_task_dataset,
)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["sst2", "mnli", "qnli", "cola", "rte"])
    parser.add_argument("--adapter-dir", "--adapter_dir", dest="adapter_dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    parser.add_argument("--output-path", "--output_path", dest="output_path", default=None)
    return parser


def ensure_padding(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.bos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer must define a pad, eos, unk, or bos token for batching.")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False


def maybe_select_subset(dataset, max_samples: int | None):
    if max_samples is None:
        return dataset
    return dataset.select(range(min(len(dataset), max_samples)))


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Layout written by `scripts/merge_task_arithmetic.copy_classifier_heads` and
# mirrored in every merge script. Each source task's trained classifier head is
# saved under `<merged_adapter>/classifier_heads/<task>.safetensors` with a
# single tensor keyed by `score.weight`. At eval time we restore the head that
# matches `task` onto the loaded model so the merged adapter keeps the trained
# classifier instead of relying on a random init.
CLASSIFIER_HEAD_SUBDIR = "classifier_heads"
CLASSIFIER_HEAD_TENSOR_KEY = "score.weight"


def _restore_classifier_head(
    model,
    adapter_path: Path,
    task: str,
    num_labels: int,
) -> dict:
    """Copy the saved per-task classifier head onto `model.base_model.model.score`.

    Returns a diagnostic dict describing the restoration (recorded in the eval
    metrics for provenance). If no saved head exists the merged adapter predates
    the classifier-head preservation change — in that case we fall back to the
    random init already seeded in `evaluate_adapter`.
    """
    head_path = adapter_path / CLASSIFIER_HEAD_SUBDIR / f"{task}.safetensors"
    if not head_path.exists():
        return {
            "restored": False,
            "reason": "no_classifier_head_file",
            "path": str(head_path),
        }

    head_state = load_file(str(head_path))
    if CLASSIFIER_HEAD_TENSOR_KEY not in head_state:
        return {
            "restored": False,
            "reason": "missing_score_weight_key",
            "path": str(head_path),
            "available_keys": sorted(head_state.keys()),
        }
    head_weight = head_state[CLASSIFIER_HEAD_TENSOR_KEY]

    target_module = model.base_model.model.score
    existing_weight = target_module.weight
    hidden_size = int(existing_weight.shape[1])
    expected_shape = (num_labels, hidden_size)
    if tuple(head_weight.shape) != expected_shape:
        raise ValueError(
            f"Classifier head shape mismatch for task {task!r}: "
            f"got {tuple(head_weight.shape)}, expected {expected_shape} "
            f"(from {head_path})."
        )

    device = existing_weight.device
    is_bnb_4bit = False
    try:
        import bitsandbytes as bnb  # type: ignore

        is_bnb_4bit = isinstance(target_module, bnb.nn.Linear4bit)
    except Exception:
        is_bnb_4bit = False

    if is_bnb_4bit:
        # The score head is tiny (num_labels x hidden_size) but
        # `load_in_4bit=True` may have replaced it with Linear4bit. Quantising the
        # trained head to NF4 would lose the very signal we are trying to
        # preserve, so we swap in a full-precision nn.Linear. Matmul with the
        # transformer's fp16 hidden states still works because PyTorch promotes
        # the Linear weight to match the input dtype.
        replacement_dtype = torch.float16
        replacement = nn.Linear(
            in_features=hidden_size,
            out_features=num_labels,
            bias=target_module.bias is not None,
            device=device,
            dtype=replacement_dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(head_weight.to(device=device, dtype=replacement_dtype))
            if replacement.bias is not None:
                replacement.bias.zero_()
        model.base_model.model.score = replacement
        restored_dtype = str(replacement_dtype)
    else:
        with torch.no_grad():
            target_module.weight.copy_(head_weight.to(device=device, dtype=existing_weight.dtype))
        restored_dtype = str(existing_weight.dtype)

    return {
        "restored": True,
        "path": str(head_path.relative_to(adapter_path)).replace("\\", "/"),
        "shape": list(head_weight.shape),
        "saved_dtype": str(head_weight.dtype),
        "runtime_dtype": restored_dtype,
        "replaced_quantized_score": is_bnb_4bit,
    }


def predicted_labels_from_output(predictions) -> np.ndarray:
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    return np.asarray(np.argmax(logits, axis=-1), dtype=np.int64)


def summarize_prediction_distribution(predicted_labels: np.ndarray, num_labels: int) -> dict:
    counts = np.bincount(predicted_labels, minlength=num_labels).astype(int)
    total = int(counts.sum())
    dominant_class = int(np.argmax(counts)) if total else None
    dominant_fraction = (float(counts[dominant_class]) / float(total)) if total else None
    return {
        "class_counts": counts.tolist(),
        "total_predictions": total,
        "dominant_class": dominant_class,
        "dominant_fraction": dominant_fraction,
        "degenerate_prediction": bool(dominant_fraction is not None and dominant_fraction > 0.95),
    }


def evaluate_adapter(
    task: str,
    adapter_dir: str,
    model_name: str | None = None,
    max_length: int | None = None,
    batch_size: int = 32,
    max_eval_samples: int | None = None,
) -> dict:
    adapter_path = Path(adapter_dir)
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    base_model_name = model_name or peft_config.base_model_name_or_path
    task_config = get_task_config(task)
    shared_cache_root = resolve_shared_eval_cache_root()
    dataset_cache_dir = str(shared_cache_root / "datasets") if shared_cache_root is not None else None
    metrics_cache_dir = str(shared_cache_root / "metrics") if shared_cache_root is not None else None
    local_files_only = env_flag("LORA_MERGE_LOCAL_FILES_ONLY") or env_flag("HF_HUB_OFFLINE")

    effective_max_length = max_length if max_length is not None else 256
    tokenizer_id = f"{base_model_name}@{effective_max_length}"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=str(shared_cache_root / "transformers") if shared_cache_root is not None else None,
        local_files_only=local_files_only,
    )
    dataset_dict = load_task_dataset(task, cache_dir=dataset_cache_dir)
    tokenized = tokenize_task_dataset(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        task_name=task,
        max_length=effective_max_length,
        cache_root=shared_cache_root,
        tokenizer_id=tokenizer_id,
    )
    eval_dataset = maybe_select_subset(tokenized[task_config.validation_split], max_eval_samples)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    eval_seed = _resolve_eval_seed()
    set_seed(eval_seed)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=task_config.num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=str(shared_cache_root / "transformers") if shared_cache_root is not None else None,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    classifier_head_restoration = _restore_classifier_head(
        model=model,
        adapter_path=adapter_path,
        task=task,
        num_labels=task_config.num_labels,
    )
    ensure_padding(tokenizer, model)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(adapter_path / "standalone_eval_tmp"),
            per_device_eval_batch_size=batch_size,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            fp16=True,
            remove_unused_columns=True,
        ),
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=build_compute_metrics_fn(task, cache_dir=metrics_cache_dir),
    )

    prediction_output = trainer.predict(eval_dataset, metric_key_prefix="eval")
    metrics = dict(prediction_output.metrics)
    if task == "cola":
        predicted_labels = predicted_labels_from_output(prediction_output.predictions)
        metrics["prediction_distribution"] = summarize_prediction_distribution(
            predicted_labels=predicted_labels,
            num_labels=task_config.num_labels,
        )
    metrics["eval_samples"] = len(eval_dataset)
    metrics["task"] = task
    metrics["adapter_dir"] = str(adapter_path)
    metrics["model_name"] = base_model_name
    metrics["classifier_head_init_seed"] = eval_seed
    metrics["classifier_head_restoration"] = classifier_head_restoration
    return to_jsonable(metrics)


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = evaluate_adapter(
        task=args.task,
        adapter_dir=args.adapter_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
    )
    output_path = Path(args.output_path) if args.output_path else Path(args.adapter_dir) / "standalone_eval_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved standalone metrics to {output_path}")


if __name__ == "__main__":
    main()
