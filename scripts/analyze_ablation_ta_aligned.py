"""Aggregate Step 4.2 (Experiment 12) aligned Task Arithmetic results.

Reads the per-variant JSONs produced by ``merge_ta_aligned.py`` under
``results/ablation_ta_aligned/{variant_alias}.json`` and writes a single
summary ``results/ablation_ta_aligned/summary.json`` that captures:

- Per-task primary metric values at the restored-head Task Arithmetic lambda
  recorded in ``results/best_hparams.json`` (unless ``--lambda`` overrides it).
- Average primary score per variant, using the same task-metric map as
  ``analyze_ablation_N.py`` / ``analyze_week3_sweep.py``.
- Delta versus the plain-TA reference row loaded from
  ``results/best_hparams.json`` (``methods.task_arithmetic``), which is the
  row in the main results table Experiment 12 asks the two new rows to be
  compared against.

The summary is deliberately compact: it's meant to be the single artifact the
dissertation results chapter appends two rows from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import get_task_config

TASKS_ALL: Tuple[str, ...] = ("sst2", "mnli", "qnli", "cola", "rte")


def extract_primary_metric(task: str, task_metrics: Dict[str, object]) -> Tuple[str, float]:
    metric_name = get_task_config(task).metric_for_best_model
    for candidate in (f"eval_{metric_name}", metric_name):
        if candidate in task_metrics:
            return metric_name, float(task_metrics[candidate])  # type: ignore[arg-type]
    raise KeyError(
        f"Primary metric '{metric_name}' not found for task '{task}' in evaluation payload."
    )


def load_variant_run(result_path: Path, lambda_value: float) -> Dict[str, object]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if not runs:
        raise ValueError(f"No runs recorded in {result_path}")

    matching = [
        run for run in runs if abs(float(run.get("lambda", float("nan"))) - lambda_value) < 1e-8
    ]
    if not matching:
        raise ValueError(
            f"No run with lambda={lambda_value} found in {result_path}; "
            f"available lambdas: {[run.get('lambda') for run in runs]}"
        )
    run = matching[0]
    evaluation = run.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"No 'evaluation' block in {result_path} for lambda={lambda_value}")

    per_task: Dict[str, Dict[str, object]] = {}
    for task in TASKS_ALL:
        task_metrics = evaluation.get(task)
        if not isinstance(task_metrics, dict):
            raise ValueError(f"Task '{task}' missing from evaluation block in {result_path}")
        metric_name, metric_value = extract_primary_metric(task, task_metrics)
        per_task[task] = {"metric": metric_name, "value": metric_value}

    values = [float(entry["value"]) for entry in per_task.values()]
    average_primary_score = sum(values) / len(values)

    return {
        "variant_label": payload.get("variant_label"),
        "result_path": str(result_path),
        "lambda": lambda_value,
        "b_weight_alpha": payload.get("b_weight_alpha"),
        "normalise_a_factors": payload.get("gpa", {}).get("normalise_a_factors"),
        "average_primary_score": average_primary_score,
        "per_task_primary": per_task,
    }


def load_plain_ta_reference(best_hparams_path: Path) -> Dict[str, object]:
    payload = json.loads(best_hparams_path.read_text(encoding="utf-8"))
    task_arithmetic = payload.get("methods", {}).get("task_arithmetic")
    if task_arithmetic is None:
        raise ValueError(f"'task_arithmetic' block not found in {best_hparams_path}")
    return {
        "variant_label": "Task Arithmetic (unaligned)",
        "result_path": str(best_hparams_path),
        "lambda": task_arithmetic.get("hyperparameters", {}).get("lambda"),
        "average_primary_score": task_arithmetic.get("average_primary_score"),
        "per_task_primary": task_arithmetic.get("primary_metrics", {}),
    }


def delta_against_reference(
    variant: Dict[str, object],
    reference: Dict[str, object],
) -> Dict[str, object]:
    per_task_delta: Dict[str, float] = {}
    for task in TASKS_ALL:
        variant_value = float(variant["per_task_primary"][task]["value"])  # type: ignore[index]
        reference_entry = reference["per_task_primary"].get(task, {})  # type: ignore[attr-defined]
        reference_value = float(reference_entry.get("value", float("nan")))
        per_task_delta[task] = variant_value - reference_value

    return {
        "reference_variant_label": reference.get("variant_label"),
        "average_primary_score_delta": float(variant["average_primary_score"])
        - float(reference["average_primary_score"]),  # type: ignore[arg-type]
        "per_task_delta": per_task_delta,
        "positive_task_count": sum(1 for delta in per_task_delta.values() if delta > 0.0),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results" / "ablation_ta_aligned"),
    )
    parser.add_argument(
        "--variants",
        default="gpa_aligned_ta,enhanced_gpa_aligned_ta",
        help=(
            "Comma-separated variant aliases. Each alias must correspond to a file "
            "'<results-root>/<alias>.json' produced by merge_ta_aligned.py."
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_value",
        type=float,
        default=None,
        help="Lambda at which to pick each variant's run (default: Task Arithmetic lambda from best_hparams).",
    )
    parser.add_argument(
        "--best-hparams-path",
        default=str(PROJECT_ROOT / "results" / "best_hparams.json"),
        help="Path to the Week 3 best-hparams summary used to load the unaligned-TA reference row.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Override the default summary output path (defaults to <results-root>/summary.json).",
    )
    return parser


def parse_csv(value: str) -> List[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def main() -> None:
    args = build_arg_parser().parse_args()
    results_root = Path(args.results_root)
    variants = parse_csv(args.variants)
    best_hparams_path = Path(args.best_hparams_path)

    reference = load_plain_ta_reference(best_hparams_path)
    lambda_value = args.lambda_value
    if lambda_value is None:
        lambda_value = float(reference["lambda"])

    rows: Dict[str, Dict[str, object]] = {}
    for alias in variants:
        result_path = results_root / f"{alias}.json"
        if not result_path.exists():
            raise FileNotFoundError(
                f"Expected variant result file not found: {result_path}. "
                "Run scripts/merge_ta_aligned.py for this variant first."
            )
        variant_row = load_variant_run(result_path, lambda_value)
        variant_row["delta_vs_task_arithmetic"] = delta_against_reference(variant_row, reference)
        rows[alias] = variant_row

    summary: Dict[str, object] = {
        "step": "step_4_2_experiment_12_task_arithmetic_in_aligned_space",
        "results_root": str(results_root),
        "lambda": lambda_value,
        "reference": reference,
        "variants": rows,
    }

    summary_path = Path(args.summary_path) if args.summary_path else results_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved aligned-TA summary to {summary_path}")

    print()
    print(f"Aligned Task Arithmetic rows at lambda={lambda_value}:")
    header = ["variant", "avg_primary", "delta_vs_TA", "pos_tasks"]
    print(
        f"{header[0]:<36} | {header[1]:>11} | {header[2]:>11} | {header[3]:>9}"
    )
    print("-" * (36 + 3 + 11 + 3 + 11 + 3 + 9))
    reference_row = (
        "Task Arithmetic (unaligned, reference)",
        reference["average_primary_score"],
        0.0,
        0,
    )
    print(
        f"{reference_row[0]:<36} | {reference_row[1]:>11.4f} | {reference_row[2]:>11.4f} | {reference_row[3]:>9}"
    )
    for alias, row in rows.items():
        delta = row["delta_vs_task_arithmetic"]
        print(
            f"{alias:<36} | "
            f"{float(row['average_primary_score']):>11.4f} | "
            f"{float(delta['average_primary_score_delta']):>+11.4f} | "
            f"{int(delta['positive_task_count']):>9}"
        )


if __name__ == "__main__":
    main()
