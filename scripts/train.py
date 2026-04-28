"""Train a single GLUE QLoRA adapter on the single RTX 6000 Ada GPU."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import accelerate
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import build_compute_metrics_fn, get_task_config, load_task_dataset, tokenize_task_dataset


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_modules_to_save(model) -> list[str]:
    modules = []
    for module_name in ("score", "classifier"):
        if hasattr(model, module_name):
            modules.append(module_name)
    return list(dict.fromkeys(modules))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["sst2", "mnli", "qnli", "cola", "rte"])
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-batch", "--per_device_batch", dest="per_device_batch", type=int, default=32)
    parser.add_argument("--grad-accum", "--grad_accum", dest="grad_accum", type=int, default=1)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=256)
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", type=int, default=500)
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", type=int, default=None)
    parser.add_argument("--logging-steps", "--logging_steps", dest="logging_steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", "--max_train_samples", dest="max_train_samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", "--report_to", dest="report_to", default="none")
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


def count_trainable_parameters(model) -> dict:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def parse_version_tuple(version: str) -> tuple[int, ...]:
    numeric_parts = []
    for part in version.split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numeric_parts.append(int(digits))
    return tuple(numeric_parts)


def main() -> None:
    args = build_arg_parser().parse_args()
    task_config = get_task_config(args.task)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_DISABLED", "true")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset_dict = load_task_dataset(args.task)
    tokenized = tokenize_task_dataset(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        task_name=args.task,
        max_length=args.max_length,
    )

    train_dataset = maybe_select_subset(tokenized["train"], args.max_train_samples)
    eval_dataset = maybe_select_subset(tokenized[task_config.validation_split], args.max_eval_samples)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=task_config.num_labels,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    ensure_padding(tokenizer, model)
    model = prepare_model_for_kbit_training(model)
    gradient_checkpointing_signature = inspect.signature(model.gradient_checkpointing_enable)
    if "gradient_checkpointing_kwargs" in gradient_checkpointing_signature.parameters:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable()
    modules_to_save = get_modules_to_save(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=DEFAULT_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics_fn(args.task)
    save_steps = args.save_steps if args.save_steps is not None else args.eval_steps
    if save_steps % args.eval_steps != 0:
        raise ValueError(
            f"save_steps ({save_steps}) must be a round multiple of eval_steps ({args.eval_steps}) "
            "when load_best_model_at_end=True."
        )

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_batch,
        "per_device_eval_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": save_steps,
        "logging_steps": args.logging_steps,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": task_config.metric_for_best_model,
        "greater_is_better": task_config.greater_is_better,
        "fp16": True,
        "bf16": False,
        "optim": "paged_adamw_8bit",
        "report_to": args.report_to,
        "remove_unused_columns": True,
        "save_total_limit": 2,
        "seed": args.seed,
    }
    accelerate_version = accelerate.__version__
    if parse_version_tuple(accelerate_version) >= (1, 1, 0):
        training_args_kwargs["data_seed"] = args.seed

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
    eval_metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    metadata = {
        "task": args.task,
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "task_config": task_config.__dict__,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "training_args": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "eval_steps": args.eval_steps,
            "save_steps": save_steps,
            "max_steps": args.max_steps,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
        },
        "accelerate_version": accelerate_version,
        "modules_to_save": modules_to_save,
        "parameter_counts": count_trainable_parameters(model),
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "eval_metrics.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
