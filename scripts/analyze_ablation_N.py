"""Aggregate Step 4.1 (Experiment 11) per-configuration merge results.

Reads the per-subset JSONs produced by ``run_ablation_N.py`` under
``results/ablation_N/N_{N}/{method}/{subset_id}.json`` plus the four N = 5
restored-head argmax rows recorded in ``results/best_hparams.json`` and
writes a single aggregated summary to ``results/ablation_N/summary.json``.

For each (N, method) cell we compute:
- Per-subset primary metric values, averaged across the tasks in that
  subset to form an ``average_primary_score`` for that subset.
- The mean and standard deviation of those per-subset averages across the
  subsets selected at that N.
- Optional per-task statistics broken down by which subsets included a
  given task, to support follow-up analysis in the Discussion chapter.

The summary is deliberately verbose: readers of the analysis can recover
the full per-subset numbers without rerunning anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import get_task_config

TASKS_ALL: Tuple[str, ...] = ("sst2", "mnli", "qnli", "cola", "rte")


def parse_csv(value: str) -> List[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def extract_primary_metric(task: str, task_metrics: Dict[str, object]) -> Tuple[str, float]:
    metric_name = get_task_config(task).metric_for_best_model
    for candidate in (f"eval_{metric_name}", metric_name):
        if candidate in task_metrics:
            return metric_name, float(task_metrics[candidate])  # type: ignore[arg-type]
    raise KeyError(
        f"Primary metric '{metric_name}' not found for task '{task}' in evaluation payload."
    )


def load_subset_evaluation(result_path: Path, subset_tasks: Sequence[str]) -> Dict[str, Dict[str, object]]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    runs = payload.get("runs") or []
    if not runs:
        raise ValueError(f"No runs recorded in {result_path}")
    run = runs[0]
    evaluation = run.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"No 'evaluation' block present in {result_path}; rerun without --skip-eval")

    per_task: Dict[str, Dict[str, object]] = {}
    for task in subset_tasks:
        task_metrics = evaluation.get(task)
        if not isinstance(task_metrics, dict):
            raise ValueError(f"Task '{task}' missing from evaluation block in {result_path}")
        metric_name, metric_value = extract_primary_metric(task, task_metrics)
        per_task[task] = {"metric": metric_name, "value": metric_value}
    return per_task


def average_primary_score(per_task: Dict[str, Dict[str, object]]) -> float:
    values = [float(entry["value"]) for entry in per_task.values()]  # type: ignore[index]
    if not values:
        raise ValueError("Cannot average primary score across zero tasks")
    return sum(values) / len(values)


def mean_std(values: Iterable[float]) -> Dict[str, object]:
    value_list = [float(value) for value in values]
    if not value_list:
        return {"values": [], "mean": None, "std": None, "n": 0}
    mean = statistics.fmean(value_list)
    std = statistics.stdev(value_list) if len(value_list) > 1 else 0.0
    return {"values": value_list, "mean": mean, "std": std, "n": len(value_list)}


def collect_cell(
    *,
    subsets_dir: Path,
    method_alias: str,
    N: int,
) -> List[Dict[str, object]]:
    method_dir = subsets_dir / f"N_{N}" / method_alias
    if not method_dir.exists():
        raise FileNotFoundError(
            f"Expected directory {method_dir} not found. Run scripts/run_ablation_N.py first."
        )

    cell_entries: List[Dict[str, object]] = []
    for result_path in sorted(method_dir.glob("*.json")):
        subset_id = result_path.stem
        subset_tasks = tuple(subset_id.split("__"))
        if len(subset_tasks) != N:
            raise ValueError(
                f"Subset file {result_path} encodes {len(subset_tasks)} tasks "
                f"but was found under N_{N}/"
            )
        per_task = load_subset_evaluation(result_path, subset_tasks)
        avg_score = average_primary_score(per_task)
        cell_entries.append(
            {
                "subset_id": subset_id,
                "tasks": list(subset_tasks),
                "result_path": str(result_path),
                "average_primary_score": avg_score,
                "per_task_primary": per_task,
            }
        )
    return cell_entries


def resolve_best_hparams_entry(payload: Dict[str, object], method_alias: str) -> Dict[str, object]:
    methods = payload.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("best_hparams payload is missing a 'methods' block")
    if method_alias == "gpa_dgpa_saties_wb_0p5":
        gpa_variants = methods.get("gpa_variants", {})
        if not isinstance(gpa_variants, dict) or method_alias not in gpa_variants:
            raise ValueError(
                "best_hparams payload is missing methods.gpa_variants.gpa_dgpa_saties_wb_0p5"
            )
        entry = gpa_variants[method_alias]
    else:
        entry = methods.get(method_alias)
    if not isinstance(entry, dict):
        raise ValueError(f"best_hparams payload is missing methods.{method_alias}")
    return entry


def resolve_source_result_path(
    best_hparams_path: Path,
    best_hparams_payload: Dict[str, object],
    entry: Dict[str, object],
) -> str:
    source_root = best_hparams_payload.get("source_root")
    source_path = entry.get("source_path")
    if isinstance(source_root, str) and isinstance(source_path, str):
        source_root_path = Path(source_root)
        if not source_root_path.is_absolute():
            source_root_path = PROJECT_ROOT / source_root_path
        return str(source_root_path / source_path)
    return str(best_hparams_path)


def load_n5_entry(
    *,
    best_hparams_path: Path,
    method_alias: str,
) -> Dict[str, object]:
    best_hparams_payload = json.loads(best_hparams_path.read_text(encoding="utf-8"))
    entry = resolve_best_hparams_entry(best_hparams_payload, method_alias)
    per_task = entry.get("primary_metrics", {})
    if not isinstance(per_task, dict):
        raise ValueError(f"best_hparams entry for '{method_alias}' is missing primary_metrics")
    avg_score = entry.get("average_primary_score")
    if avg_score is None:
        raise ValueError(f"best_hparams entry for '{method_alias}' is missing average_primary_score")
    subset_id = "__".join(TASKS_ALL)
    return {
        "subset_id": subset_id,
        "tasks": list(TASKS_ALL),
        "result_path": resolve_source_result_path(best_hparams_path, best_hparams_payload, entry),
        "average_primary_score": float(avg_score),
        "per_task_primary": per_task,
    }


def summarise_cell(entries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    scores = [float(entry["average_primary_score"]) for entry in entries]
    per_task_grouping: Dict[str, List[float]] = {}
    for entry in entries:
        for task, task_block in entry["per_task_primary"].items():  # type: ignore[attr-defined]
            per_task_grouping.setdefault(task, []).append(float(task_block["value"]))  # type: ignore[index]

    per_task_summary: Dict[str, Dict[str, object]] = {}
    for task, values in per_task_grouping.items():
        per_task_summary[task] = {
            "metric": get_task_config(task).metric_for_best_model,
            **mean_std(values),
        }

    return {
        "subset_count": len(entries),
        "average_primary_score": mean_std(scores),
        "per_task_primary": per_task_summary,
        "subsets": list(entries),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results" / "ablation_N"),
    )
    parser.add_argument(
        "--best-hparams-path",
        default=str(PROJECT_ROOT / "results" / "best_hparams.json"),
    )
    parser.add_argument(
        "--methods",
        default="gpa_baseline,gpa_dgpa_saties_wb_0p5,ties,lr_knots",
    )
    parser.add_argument(
        "--n-values",
        default="2,3,4,5",
        help="N values to include in the summary. N=5 always sources from best_hparams.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Override the default summary output path (defaults to <results-root>/summary.json).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results_root = Path(args.results_root)
    best_hparams_path = Path(args.best_hparams_path)
    methods = parse_csv(args.methods)
    n_values = [int(piece) for piece in parse_csv(args.n_values)]

    manifest_path = results_root / "run_manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    summary: Dict[str, object] = {
        "step": "step_4_1_experiment_11_vary_N",
        "results_root": str(results_root),
        "best_hparams_path": str(best_hparams_path),
        "methods": methods,
        "n_values": n_values,
        "run_manifest": manifest,
        "cells": {},
    }

    for method_alias in methods:
        method_summary: Dict[str, object] = {}
        for N in n_values:
            if N == 5:
                entries = [
                    load_n5_entry(
                        best_hparams_path=best_hparams_path,
                        method_alias=method_alias,
                    )
                ]
            else:
                entries = collect_cell(
                    subsets_dir=results_root,
                    method_alias=method_alias,
                    N=N,
                )
            method_summary[str(N)] = summarise_cell(entries)
        summary["cells"][method_alias] = method_summary  # type: ignore[index]

    summary_path = Path(args.summary_path) if args.summary_path else results_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved ablation-N summary to {summary_path}")

    print()
    print("Per-(N, method) mean average primary score:")
    header_N = [str(N) for N in n_values]
    print("method".ljust(32) + " | " + " | ".join(col.center(10) for col in header_N))
    print("-" * (32 + 3 + (10 + 3) * len(header_N)))
    for method_alias in methods:
        row_cells = []
        for N in n_values:
            cell = summary["cells"][method_alias][str(N)]  # type: ignore[index]
            mean_value = cell["average_primary_score"]["mean"]
            row_cells.append(f"{mean_value:.4f}" if mean_value is not None else "-")
        print(method_alias.ljust(32) + " | " + " | ".join(value.center(10) for value in row_cells))


if __name__ == "__main__":
    main()
