"""Warm shared Hugging Face caches for repeated evaluation runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import build_compute_metrics_fn, load_task_dataset, resolve_shared_eval_cache_root, tokenize_task_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tasks", nargs="+", default=["sst2", "mnli", "qnli", "cola", "rte"])
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=256)
    parser.add_argument("--local-files-only", "--local_files_only", dest="local_files_only", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cache_root = resolve_shared_eval_cache_root()
    dataset_cache_dir = str(cache_root / "datasets") if cache_root is not None else None
    metrics_cache_dir = str(cache_root / "metrics") if cache_root is not None else None
    transformer_cache_dir = str(cache_root / "transformers") if cache_root is not None else None

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=transformer_cache_dir,
        local_files_only=args.local_files_only,
    )
    tokenizer_id = f"{args.model_name}@{args.max_length}"

    for task in args.tasks:
        dataset = load_task_dataset(task, cache_dir=dataset_cache_dir)
        tokenize_task_dataset(
            dataset,
            tokenizer=tokenizer,
            task_name=task,
            max_length=args.max_length,
            cache_root=cache_root,
            tokenizer_id=tokenizer_id,
        )
        build_compute_metrics_fn(task, cache_dir=metrics_cache_dir)
        print(f"Warmed cache for {task}")


if __name__ == "__main__":
    main()
