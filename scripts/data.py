"""GLUE dataset helpers for QLoRA training."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    dataset_name: str
    dataset_subset: str
    text_key_1: str
    text_key_2: Optional[str]
    num_labels: int
    validation_split: str
    metric_name: str
    metric_for_best_model: str
    greater_is_better: bool = True


TASK_CONFIG: Dict[str, TaskConfig] = {
    "sst2": TaskConfig(
        task_name="sst2",
        dataset_name="glue",
        dataset_subset="sst2",
        text_key_1="sentence",
        text_key_2=None,
        num_labels=2,
        validation_split="validation",
        metric_name="glue",
        metric_for_best_model="accuracy",
    ),
    "mnli": TaskConfig(
        task_name="mnli",
        dataset_name="glue",
        dataset_subset="mnli",
        text_key_1="premise",
        text_key_2="hypothesis",
        num_labels=3,
        validation_split="validation_matched",
        metric_name="glue",
        metric_for_best_model="accuracy",
    ),
    "qnli": TaskConfig(
        task_name="qnli",
        dataset_name="glue",
        dataset_subset="qnli",
        text_key_1="question",
        text_key_2="sentence",
        num_labels=2,
        validation_split="validation",
        metric_name="glue",
        metric_for_best_model="accuracy",
    ),
    "cola": TaskConfig(
        task_name="cola",
        dataset_name="glue",
        dataset_subset="cola",
        text_key_1="sentence",
        text_key_2=None,
        num_labels=2,
        validation_split="validation",
        metric_name="glue",
        metric_for_best_model="matthews_correlation",
    ),
    "rte": TaskConfig(
        task_name="rte",
        dataset_name="glue",
        dataset_subset="rte",
        text_key_1="sentence1",
        text_key_2="sentence2",
        num_labels=2,
        validation_split="validation",
        metric_name="glue",
        metric_for_best_model="accuracy",
    ),
}


def get_task_config(task_name: str) -> TaskConfig:
    key = task_name.lower()
    if key not in TASK_CONFIG:
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {sorted(TASK_CONFIG)}")
    return TASK_CONFIG[key]


def resolve_shared_eval_cache_root() -> Path | None:
    raw_value = os.environ.get("LORA_MERGE_EVAL_CACHE_DIR")
    if not raw_value:
        return None
    cache_root = Path(raw_value)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _stable_cache_key(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def build_tokenized_cache_file_names(
    dataset_dict: DatasetDict,
    *,
    task_name: str,
    max_length: int,
    tokenizer_id: str,
    cache_root: Path | None,
) -> Dict[str, str] | None:
    if cache_root is None:
        return None

    tokenized_root = cache_root / "tokenized"
    tokenized_root.mkdir(parents=True, exist_ok=True)
    cache_key = _stable_cache_key(task_name.lower(), tokenizer_id, str(max_length))
    return {
        split_name: str(tokenized_root / f"{task_name.lower()}_{cache_key}_{split_name}.arrow")
        for split_name in dataset_dict.keys()
    }


def load_task_dataset(task_name: str, *, cache_dir: str | None = None) -> DatasetDict:
    config = get_task_config(task_name)
    return load_dataset(config.dataset_name, config.dataset_subset, cache_dir=cache_dir)


def build_tokenize_function(tokenizer, task_name: str, max_length: int) -> Callable[[Dict[str, list]], Dict[str, list]]:
    config = get_task_config(task_name)

    def tokenize_function(batch: Dict[str, list]) -> Dict[str, list]:
        if config.text_key_2 is None:
            return tokenizer(
                batch[config.text_key_1],
                truncation=True,
                max_length=max_length,
            )
        return tokenizer(
            batch[config.text_key_1],
            batch[config.text_key_2],
            truncation=True,
            max_length=max_length,
        )

    return tokenize_function


def tokenize_task_dataset(
    dataset_dict: DatasetDict,
    tokenizer,
    task_name: str,
    max_length: int,
    *,
    cache_root: Path | None = None,
    tokenizer_id: str = "default_tokenizer",
) -> DatasetDict:
    tokenize_function = build_tokenize_function(tokenizer=tokenizer, task_name=task_name, max_length=max_length)
    cache_file_names = build_tokenized_cache_file_names(
        dataset_dict,
        task_name=task_name,
        max_length=max_length,
        tokenizer_id=tokenizer_id,
        cache_root=cache_root,
    )
    return dataset_dict.map(
        tokenize_function,
        batched=True,
        desc=f"Tokenizing {task_name}",
        load_from_cache_file=True,
        cache_file_names=cache_file_names,
    )


def build_compute_metrics_fn(task_name: str, *, cache_dir: str | None = None) -> Callable:
    config = get_task_config(task_name)
    metric = evaluate.load(config.metric_name, config.dataset_subset, cache_dir=cache_dir)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted_labels = np.argmax(predictions, axis=-1)
        return metric.compute(predictions=predicted_labels, references=labels)

    return compute_metrics
