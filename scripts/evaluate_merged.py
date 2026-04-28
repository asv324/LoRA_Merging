"""Unified evaluation for merged LoRA adapters."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapter_eval import evaluate_adapter
from scripts.data import get_task_config
from scripts.merge_task_arithmetic import (
    CLASSIFIER_HEAD_MANIFEST_FILENAME,
    CLASSIFIER_HEAD_SUBDIR,
    TASKS,
    TOKENIZER_ASSETS,
    to_jsonable,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", "--adapter_dir", dest="adapter_dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument(
        "--lambdas",
        default=None,
        help="Optional comma-separated lambda values to evaluate. If omitted, evaluates the adapter as saved.",
    )
    parser.add_argument(
        "--stored-lambda",
        "--stored_lambda",
        dest="stored_lambda",
        type=float,
        default=None,
        help="Lambda baked into the saved adapter. Inferred from merge_metadata.json when available.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    parser.add_argument("--output-path", "--output_path", dest="output_path", default=None)
    return parser


def parse_lambda_values(raw_value: str | None) -> List[float] | None:
    if raw_value is None:
        return None
    values = []
    for piece in raw_value.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0:
            raise ValueError("lambda values must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("at least one lambda value is required")
    return values


def read_optional_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def infer_trained_tasks(adapter_dir: Path, explicit_tasks: Sequence[str] | None) -> List[str]:
    if explicit_tasks:
        return list(explicit_tasks)
    metadata = read_optional_json(adapter_dir / "merge_metadata.json") or read_optional_json(adapter_dir / "alignment_metadata.json")
    if metadata and metadata.get("source_tasks"):
        return list(metadata["source_tasks"])
    return list(TASKS)


def infer_stored_lambda(adapter_dir: Path, cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    metadata = read_optional_json(adapter_dir / "merge_metadata.json")
    if metadata and metadata.get("lambda") is not None:
        return float(metadata["lambda"])
    return 1.0


def scale_lora_a_state_dict(state_dict: Dict[str, object], scale_factor: float) -> Dict[str, object]:
    scaled = {}
    for key, value in state_dict.items():
        if key.endswith(".lora_A.weight"):
            scaled[key] = value.clone().mul(scale_factor)
        elif hasattr(value, "clone"):
            scaled[key] = value.clone()
        else:
            scaled[key] = value
    return scaled


def write_scaled_adapter_copy(source_adapter_dir: Path, output_dir: Path, target_lambda: float, stored_lambda: float) -> Path:
    if stored_lambda == 0.0:
        if target_lambda == 0.0:
            scale_factor = 1.0
        else:
            raise ValueError("Cannot rescale a stored lambda of 0.0 to a non-zero target lambda.")
    else:
        scale_factor = target_lambda / stored_lambda

    state_dict = load_file(str(source_adapter_dir / "adapter_model.safetensors"))
    scaled_state_dict = scale_lora_a_state_dict(state_dict, scale_factor=scale_factor)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(scaled_state_dict), str(output_dir / "adapter_model.safetensors"))
    shutil.copy2(source_adapter_dir / "adapter_config.json", output_dir / "adapter_config.json")
    for asset_name in TOKENIZER_ASSETS:
        source_path = source_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)
    heads_dir = source_adapter_dir / CLASSIFIER_HEAD_SUBDIR
    if heads_dir.exists():
        shutil.copytree(heads_dir, output_dir / CLASSIFIER_HEAD_SUBDIR, dirs_exist_ok=True)
    manifest_path = source_adapter_dir / CLASSIFIER_HEAD_MANIFEST_FILENAME
    if manifest_path.exists():
        shutil.copy2(manifest_path, output_dir / CLASSIFIER_HEAD_MANIFEST_FILENAME)
    metadata = read_optional_json(source_adapter_dir / "merge_metadata.json")
    if metadata is not None:
        metadata["lambda"] = target_lambda
        metadata["rescaled_from_lambda"] = stored_lambda
        (output_dir / "merge_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    return output_dir


def extract_primary_metric(task: str, metrics: Dict[str, object]) -> tuple[str, float]:
    task_config = get_task_config(task)
    metric_name = task_config.metric_for_best_model
    candidate_keys = [f"eval_{metric_name}", metric_name]
    for key in candidate_keys:
        if key in metrics:
            return metric_name, float(metrics[key])
    raise KeyError(f"Primary metric '{metric_name}' not found for task '{task}'.")


def summarize_evaluations(evaluations: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    primary_metrics = {}
    for task, task_metrics in evaluations.items():
        metric_name, metric_value = extract_primary_metric(task, task_metrics)
        primary_metrics[task] = {"metric": metric_name, "value": metric_value}
    average_primary_score = sum(item["value"] for item in primary_metrics.values()) / len(primary_metrics)
    return {
        "primary_metrics": primary_metrics,
        "average_primary_score": average_primary_score,
    }


def evaluate_adapter_dir(
    adapter_dir: Path,
    *,
    tasks: Sequence[str],
    model_name: str | None,
    max_length: int | None,
    batch_size: int,
    max_eval_samples: int | None,
) -> Dict[str, Dict[str, object]]:
    evaluations = {}
    for task in tasks:
        evaluations[task] = to_jsonable(
            evaluate_adapter(
                task=task,
                adapter_dir=str(adapter_dir),
                model_name=model_name,
                max_length=max_length,
                batch_size=batch_size,
                max_eval_samples=max_eval_samples,
            )
        )
    return evaluations


def main() -> None:
    args = build_arg_parser().parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    tasks = infer_trained_tasks(adapter_dir, args.tasks)
    lambda_values = parse_lambda_values(args.lambdas)
    stored_lambda = infer_stored_lambda(adapter_dir, args.stored_lambda)
    if lambda_values is None:
        lambda_values = [stored_lambda]

    results_payload = {
        "adapter_dir": str(adapter_dir),
        "evaluated_tasks": tasks,
        "stored_lambda": stored_lambda,
        "lambda_values": lambda_values,
        "merge_metadata": read_optional_json(adapter_dir / "merge_metadata.json"),
        "runs": [],
    }

    for lambda_value in lambda_values:
        if lambda_value == stored_lambda:
            evaluations = evaluate_adapter_dir(
                adapter_dir,
                tasks=tasks,
                model_name=args.model_name,
                max_length=args.max_length,
                batch_size=args.batch_size,
                max_eval_samples=args.max_eval_samples,
            )
            run_summary = {
                "lambda": lambda_value,
                "used_rescaled_copy": False,
                "evaluated_adapter_dir": str(adapter_dir),
                "per_task": evaluations,
                "summary": summarize_evaluations(evaluations),
            }
        else:
            with tempfile.TemporaryDirectory(prefix="evaluate_merged_", dir=str(PROJECT_ROOT)) as temp_dir:
                evaluation_adapter_dir = write_scaled_adapter_copy(
                    source_adapter_dir=adapter_dir,
                    output_dir=Path(temp_dir) / "adapter",
                    target_lambda=lambda_value,
                    stored_lambda=stored_lambda,
                )
                evaluations = evaluate_adapter_dir(
                    evaluation_adapter_dir,
                    tasks=tasks,
                    model_name=args.model_name,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                    max_eval_samples=args.max_eval_samples,
                )
            run_summary = {
                "lambda": lambda_value,
                "used_rescaled_copy": True,
                "evaluated_adapter_dir": str(adapter_dir),
                "per_task": evaluations,
                "summary": summarize_evaluations(evaluations),
            }
        results_payload["runs"].append(run_summary)

    output_path = Path(args.output_path) if args.output_path else adapter_dir / "evaluation_summary.json"
    output_path.write_text(json.dumps(to_jsonable(results_payload), indent=2), encoding="utf-8")
    print(json.dumps(to_jsonable(results_payload), indent=2))
    print(f"Saved merged adapter evaluation to {output_path}")


if __name__ == "__main__":
    main()
