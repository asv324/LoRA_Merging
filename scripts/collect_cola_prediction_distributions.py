"""Collect CoLA prediction distributions from sweep result JSONs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_week3_sweep import (
    EXCLUDED_JSON_NAMES,
    GPA_VARIANT_ORDER,
    METHOD_LABELS,
    infer_method_key,
    to_jsonable,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        "--sweep_root",
        dest="sweep_root",
        default=PROJECT_ROOT / "results" / "hp_sweep_low_storage",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        default=PROJECT_ROOT / "results" / "hp_sweep" / "cola_prediction_distributions.json",
    )
    return parser


def iter_result_json_paths(root: Path) -> List[Path]:
    manifest_path = root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_files = manifest.get("result_files")
        if isinstance(result_files, list):
            return [root / relative_path for relative_path in result_files]
    return [path for path in sorted(root.rglob("*.json")) if path.name not in EXCLUDED_JSON_NAMES]


def summarize_method_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    available = [record for record in records if record["prediction_distribution_available"]]
    if not available:
        best_metric_record = max(records, key=lambda record: float(record["cola_metric"]))
        return {
            "prediction_distribution_available": False,
            "record_count": len(records),
            "non_degenerate_lambda_threshold": None,
            "best_cola_metric": best_metric_record["cola_metric"],
            "best_cola_metric_source_path": best_metric_record["source_path"],
            "rerun_required": True,
        }

    sorted_available = sorted(
        available,
        key=lambda record: (
            float(record["lambda"]) if record["lambda"] is not None else float("inf"),
            record["source_path"],
        ),
    )
    threshold_record = next((record for record in sorted_available if not record["degenerate_prediction"]), None)
    best_metric_record = max(sorted_available, key=lambda record: float(record["cola_metric"]))
    return {
        "prediction_distribution_available": True,
        "record_count": len(records),
        "available_record_count": len(available),
        "non_degenerate_lambda_threshold": threshold_record["lambda"] if threshold_record is not None else None,
        "non_degenerate_source_path": threshold_record["source_path"] if threshold_record is not None else None,
        "best_cola_metric": best_metric_record["cola_metric"],
        "best_cola_metric_source_path": best_metric_record["source_path"],
        "rerun_required": threshold_record is None,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    sweep_root = Path(args.sweep_root)

    records: List[Dict[str, object]] = []
    summary_by_method: Dict[str, List[Dict[str, object]]] = {}
    for path in iter_result_json_paths(sweep_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload.get("runs")
        if not isinstance(runs, list):
            continue
        relative_path = path.relative_to(sweep_root)
        for index, run in enumerate(runs):
            evaluation = run.get("evaluation")
            if not isinstance(evaluation, dict) or "cola" not in evaluation:
                continue
            cola_eval = evaluation["cola"]
            distribution = cola_eval.get("prediction_distribution")
            method_key = infer_method_key(relative_path, payload)
            record = {
                "source_path": (
                    relative_path.as_posix()
                    if len(runs) == 1
                    else f"{relative_path.as_posix()}#run{index}"
                ),
                "method_key": method_key,
                "display_name": METHOD_LABELS[method_key],
                "lambda": run.get("lambda"),
                "trim_percentage": run.get("trim_percentage"),
                "drop_probability": run.get("drop_probability"),
                "cola_metric": cola_eval.get("eval_matthews_correlation", cola_eval.get("matthews_correlation")),
                "prediction_distribution_available": isinstance(distribution, dict),
                "class_counts": distribution.get("class_counts") if isinstance(distribution, dict) else None,
                "total_predictions": distribution.get("total_predictions") if isinstance(distribution, dict) else None,
                "dominant_class": distribution.get("dominant_class") if isinstance(distribution, dict) else None,
                "dominant_fraction": distribution.get("dominant_fraction") if isinstance(distribution, dict) else None,
                "degenerate_prediction": distribution.get("degenerate_prediction") if isinstance(distribution, dict) else None,
            }
            records.append(record)
            summary_by_method.setdefault(method_key, []).append(record)

    ordered_summary: Dict[str, Dict[str, object]] = {}
    ordered_keys = [key for key in GPA_VARIANT_ORDER if key in summary_by_method]
    ordered_keys = ["task_arithmetic", "ties", "dare_ties", "lr_knots"] + ordered_keys
    ordered_keys = [key for key in ordered_keys if key in summary_by_method]
    for method_key in ordered_keys:
        ordered_summary[method_key] = summarize_method_records(summary_by_method[method_key])

    payload = {
        "source_root": str(sweep_root),
        "record_count": len(records),
        "prediction_distribution_available": any(record["prediction_distribution_available"] for record in records),
        "summary_by_method": ordered_summary,
        "records": records,
        "notes": [
            "Runs created before prediction-distribution logging was added will report prediction_distribution_available=false.",
            "For methods without available class counts, rerun CoLA evaluation is required to compute a true non-degenerate lambda threshold.",
        ],
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Saved CoLA prediction distributions to {output_path}")


if __name__ == "__main__":
    main()
