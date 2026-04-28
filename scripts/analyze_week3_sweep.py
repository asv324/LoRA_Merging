"""Aggregate Week 3 sweep results into summary artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import get_task_config

TASK_ORDER = ["sst2", "mnli", "qnli", "cola", "rte"]
PRIMARY_METHOD_ORDER = [
    "oracle",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "lr_knots",
    "gpa_baseline",
    "gpa_best_enhanced",
]
GPA_VARIANT_ORDER = [
    "gpa_baseline",
    "gpa_dgpa_ties",
    "gpa_dgpa_saties",
    "gpa_dgpa_saties_wb_0p5",
    "gpa_dgpa_saties_wb_1p0",
]
METHOD_LABELS = {
    "oracle": "Individual (oracle)",
    "task_arithmetic": "Task Arithmetic",
    "ties": "TIES-Merging",
    "dare_ties": "DARE + TIES",
    "lr_knots": "LR-KnOTS + TIES",
    "gpa_baseline": "GPA + TIES",
    "gpa_dgpa_ties": "dGPA + TIES",
    "gpa_dgpa_saties": "dGPA + saTIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA + saTIES + wB(0.5)",
    "gpa_dgpa_saties_wb_1p0": "dGPA + saTIES + wB(1.0)",
    "gpa_best_enhanced": "dGPA + saTIES + wB (best enhanced)",
}
GPA_VARIANT_SLUGS = {
    "baseline": "gpa_baseline",
    "dgpa_ties": "gpa_dgpa_ties",
    "dgpa_saties": "gpa_dgpa_saties",
    "dgpa_saties_wb_0p5": "gpa_dgpa_saties_wb_0p5",
    "dgpa_saties_wb_1p0": "gpa_dgpa_saties_wb_1p0",
}
EXCLUDED_JSON_NAMES = {
    "run_manifest.json",
    "best_hparams.json",
    "main_results.json",
    "enhancement_ablation.json",
    "gpa_rerun_decision.json",
    "cola_prediction_distributions.json",
}


@dataclass(frozen=True)
class SweepRecord:
    method_key: str
    display_name: str
    source_path: str
    average_primary_score: float
    primary_metrics: Dict[str, Dict[str, float | str]]
    evaluation: Dict[str, Dict[str, object]]
    lambda_value: float | None
    trim_percentage: int | None
    drop_probability: float | None
    density: float | None
    b_weight_alpha: float | None
    variant_label: str | None
    normalise_a_factors: bool | None
    scale_aware_ties: bool | None
    gpa_module_count: int | None
    gpa_modules_hitting_max_iter: int | None
    gpa_max_iter_fraction: float | None


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
    parser.add_argument(
        "--sweep-root",
        "--sweep_root",
        dest="sweep_root",
        default=PROJECT_ROOT / "results" / "hp_sweep_low_storage",
    )
    parser.add_argument(
        "--adapters-dir",
        "--adapters_dir",
        dest="adapters_dir",
        default=PROJECT_ROOT / "adapters",
    )
    parser.add_argument(
        "--best-hparams-path",
        "--best_hparams_path",
        dest="best_hparams_path",
        default=PROJECT_ROOT / "results" / "best_hparams.json",
    )
    parser.add_argument(
        "--main-results-path",
        "--main_results_path",
        dest="main_results_path",
        default=PROJECT_ROOT / "results" / "main_results.json",
    )
    parser.add_argument(
        "--enhancement-ablation-path",
        "--enhancement_ablation_path",
        dest="enhancement_ablation_path",
        default=PROJECT_ROOT / "results" / "enhancement_ablation.json",
    )
    parser.add_argument(
        "--gpa-rerun-decision-path",
        "--gpa_rerun_decision_path",
        dest="gpa_rerun_decision_path",
        default=PROJECT_ROOT / "results" / "gpa_rerun_decision.json",
    )
    return parser


def extract_primary_metric(task: str, metrics: Dict[str, object]) -> tuple[str, float]:
    metric_name = get_task_config(task).metric_for_best_model
    for candidate in (f"eval_{metric_name}", metric_name):
        if candidate in metrics:
            return metric_name, float(metrics[candidate])
    raise KeyError(f"Primary metric '{metric_name}' not found for task '{task}'.")


def compute_primary_metrics(evaluation: Dict[str, Dict[str, object]]) -> tuple[Dict[str, Dict[str, float | str]], float]:
    primary_metrics: Dict[str, Dict[str, float | str]] = {}
    for task in TASK_ORDER:
        metric_name, metric_value = extract_primary_metric(task, evaluation[task])
        primary_metrics[task] = {"metric": metric_name, "value": metric_value}
    average_primary_score = sum(float(item["value"]) for item in primary_metrics.values()) / len(TASK_ORDER)
    return primary_metrics, average_primary_score


def infer_method_key(relative_path: Path, payload: Dict[str, object]) -> str:
    top_level = relative_path.parts[0]
    if top_level == "task_arithmetic":
        return "task_arithmetic"
    if top_level == "ties":
        return "ties"
    if top_level == "dare_ties":
        return "dare_ties"
    if top_level == "lr_knots":
        return "lr_knots"
    if top_level == "gpa_ties":
        slug = relative_path.parts[1] if len(relative_path.parts) > 1 else "baseline"
        return GPA_VARIANT_SLUGS.get(slug, "gpa_baseline")

    method = payload.get("method")
    variant_label = payload.get("variant_label")
    if method == "task_arithmetic_exact_delta_space":
        return "task_arithmetic"
    if method == "ties_factor_space_raw_lora_factors":
        return "ties"
    if method == "dare_adapter_sparsification":
        return "dare_ties"
    if method == "lr_knots_factor_space_ties":
        return "lr_knots"
    if method == "gpa_ties_factor_space":
        variant_mapping = {
            None: "gpa_baseline",
            "GPA+TIES": "gpa_baseline",
            "dGPA+TIES": "gpa_dgpa_ties",
            "dGPA+saTIES": "gpa_dgpa_saties",
            "dGPA+saTIES+wB(0.5)": "gpa_dgpa_saties_wb_0p5",
            "dGPA+saTIES+wB(1)": "gpa_dgpa_saties_wb_1p0",
            "dGPA+saTIES+wB(1.0)": "gpa_dgpa_saties_wb_1p0",
        }
        return variant_mapping.get(variant_label, "gpa_baseline")
    raise ValueError(f"Could not infer method key for {relative_path}")


def build_record(relative_path: Path, payload: Dict[str, object], run: Dict[str, object]) -> SweepRecord:
    evaluation = run["evaluation"]
    primary_metrics, average_primary_score = compute_primary_metrics(evaluation)
    method_key = infer_method_key(relative_path, payload)
    display_name = METHOD_LABELS[method_key]

    gpa_module_count = None
    gpa_modules_hitting_max_iter = None
    gpa_max_iter_fraction = None
    if method_key.startswith("gpa_"):
        module_diagnostics = payload.get("gpa", {}).get("module_diagnostics", [])
        max_iter = int(payload.get("gpa", {}).get("max_iter", 100))
        gpa_module_count = len(module_diagnostics)
        gpa_modules_hitting_max_iter = sum(1 for diag in module_diagnostics if diag.get("iterations") == max_iter)
        gpa_max_iter_fraction = (
            gpa_modules_hitting_max_iter / gpa_module_count if gpa_module_count else None
        )

    return SweepRecord(
        method_key=method_key,
        display_name=display_name,
        source_path=relative_path.as_posix(),
        average_primary_score=average_primary_score,
        primary_metrics=primary_metrics,
        evaluation=evaluation,
        lambda_value=float(run["lambda"]) if run.get("lambda") is not None else None,
        trim_percentage=int(run["trim_percentage"]) if run.get("trim_percentage") is not None else None,
        drop_probability=float(run["drop_probability"]) if run.get("drop_probability") is not None else None,
        density=float(run["density"]) if run.get("density") is not None else None,
        b_weight_alpha=float(payload["b_weight_alpha"]) if payload.get("b_weight_alpha") is not None else None,
        variant_label=payload.get("variant_label"),
        normalise_a_factors=payload.get("gpa", {}).get("normalise_a_factors"),
        scale_aware_ties=payload.get("scale_aware_ties"),
        gpa_module_count=gpa_module_count,
        gpa_modules_hitting_max_iter=gpa_modules_hitting_max_iter,
        gpa_max_iter_fraction=gpa_max_iter_fraction,
    )


def iter_sweep_json_paths(sweep_root: Path) -> List[Path]:
    manifest_path = sweep_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result_files = manifest.get("result_files")
        if isinstance(result_files, list):
            return [sweep_root / relative_path for relative_path in result_files]

    return [
        path
        for path in sorted(sweep_root.rglob("*.json"))
        if path.name not in EXCLUDED_JSON_NAMES
    ]


def load_sweep_records(sweep_root: Path) -> List[SweepRecord]:
    records: List[SweepRecord] = []
    for path in iter_sweep_json_paths(sweep_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        runs = payload.get("runs")
        if not isinstance(runs, list):
            continue
        relative_path = path.relative_to(sweep_root)
        for index, run in enumerate(runs):
            if not isinstance(run, dict) or "evaluation" not in run:
                continue
            run_relative_path = relative_path if len(runs) == 1 else Path(str(relative_path) + f"#run{index}")
            records.append(build_record(run_relative_path, payload, run))
    return records


def load_oracle_metrics(adapters_dir: Path) -> Dict[str, Dict[str, object]] | None:
    metrics_by_task: Dict[str, Dict[str, object]] = {}
    for task in TASK_ORDER:
        metrics_path = adapters_dir / task / "eval_metrics.json"
        if not metrics_path.exists():
            return None
        metrics_by_task[task] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metrics_by_task


def build_oracle_record(adapters_dir: Path) -> SweepRecord | None:
    oracle_metrics = load_oracle_metrics(adapters_dir)
    if oracle_metrics is None:
        return None
    primary_metrics, average_primary_score = compute_primary_metrics(oracle_metrics)
    return SweepRecord(
        method_key="oracle",
        display_name=METHOD_LABELS["oracle"],
        source_path="adapters/*/eval_metrics.json",
        average_primary_score=average_primary_score,
        primary_metrics=primary_metrics,
        evaluation=oracle_metrics,
        lambda_value=None,
        trim_percentage=None,
        drop_probability=None,
        density=None,
        b_weight_alpha=None,
        variant_label=None,
        normalise_a_factors=None,
        scale_aware_ties=None,
        gpa_module_count=None,
        gpa_modules_hitting_max_iter=None,
        gpa_max_iter_fraction=None,
    )


def select_best_record(records: Sequence[SweepRecord]) -> SweepRecord:
    return sorted(records, key=lambda record: (-record.average_primary_score, record.source_path))[0]


def serialize_record(record: SweepRecord) -> Dict[str, object]:
    payload = {
        "method_key": record.method_key,
        "display_name": record.display_name,
        "source_path": record.source_path,
        "average_primary_score": record.average_primary_score,
        "primary_metrics": record.primary_metrics,
        "hyperparameters": {
            "lambda": record.lambda_value,
            "trim_percentage": record.trim_percentage,
            "drop_probability": record.drop_probability,
            "density": record.density,
            "b_weight_alpha": record.b_weight_alpha,
            "normalise_a_factors": record.normalise_a_factors,
            "scale_aware_ties": record.scale_aware_ties,
        },
    }
    if record.variant_label is not None:
        payload["variant_label"] = record.variant_label
    if record.gpa_module_count is not None:
        payload["gpa_convergence"] = {
            "module_count": record.gpa_module_count,
            "modules_hitting_max_iter": record.gpa_modules_hitting_max_iter,
            "max_iter_fraction": record.gpa_max_iter_fraction,
        }
    return payload


def build_best_hparams_payload(
    records: Sequence[SweepRecord],
    oracle_record: SweepRecord | None,
    *,
    source_root: Path,
) -> Dict[str, object]:
    records_by_method: Dict[str, List[SweepRecord]] = {}
    for record in records:
        records_by_method.setdefault(record.method_key, []).append(record)

    best_per_method = {method_key: select_best_record(items) for method_key, items in records_by_method.items()}
    enhanced_gpa_records = [best_per_method[key] for key in GPA_VARIANT_ORDER[1:] if key in best_per_method]
    best_enhanced = select_best_record(enhanced_gpa_records) if enhanced_gpa_records else None

    payload = {
        "source_root": str(source_root),
        "task_order": TASK_ORDER,
        "selection_metric": "average_primary_score",
        "methods": {
            "task_arithmetic": serialize_record(best_per_method["task_arithmetic"]),
            "ties": serialize_record(best_per_method["ties"]),
            "dare_ties": serialize_record(best_per_method["dare_ties"]),
            "lr_knots": serialize_record(best_per_method["lr_knots"]),
            "gpa_baseline": serialize_record(best_per_method["gpa_baseline"]),
            "gpa_variants": {
                method_key: serialize_record(best_per_method[method_key])
                for method_key in GPA_VARIANT_ORDER
                if method_key in best_per_method
            },
        },
    }
    if best_enhanced is not None:
        payload["methods"]["gpa_best_enhanced"] = serialize_record(best_enhanced)
    if oracle_record is not None:
        payload["methods"]["oracle"] = serialize_record(oracle_record)
    return payload


def build_main_results_payload(
    records: Sequence[SweepRecord],
    oracle_record: SweepRecord | None,
    *,
    source_root: Path,
) -> Dict[str, object]:
    records_by_method: Dict[str, List[SweepRecord]] = {}
    for record in records:
        records_by_method.setdefault(record.method_key, []).append(record)

    best_per_method = {method_key: select_best_record(items) for method_key, items in records_by_method.items()}
    enhanced_gpa_records = [best_per_method[key] for key in GPA_VARIANT_ORDER[1:] if key in best_per_method]
    best_enhanced = select_best_record(enhanced_gpa_records) if enhanced_gpa_records else None

    main_rows: List[SweepRecord] = []
    if oracle_record is not None:
        main_rows.append(oracle_record)
    for method_key in ("task_arithmetic", "ties", "dare_ties", "lr_knots", "gpa_baseline"):
        main_rows.append(best_per_method[method_key])
    if best_enhanced is not None:
        synthetic_best = SweepRecord(
            method_key="gpa_best_enhanced",
            display_name=METHOD_LABELS["gpa_best_enhanced"],
            source_path=best_enhanced.source_path,
            average_primary_score=best_enhanced.average_primary_score,
            primary_metrics=best_enhanced.primary_metrics,
            evaluation=best_enhanced.evaluation,
            lambda_value=best_enhanced.lambda_value,
            trim_percentage=best_enhanced.trim_percentage,
            drop_probability=best_enhanced.drop_probability,
            density=best_enhanced.density,
            b_weight_alpha=best_enhanced.b_weight_alpha,
            variant_label=best_enhanced.variant_label,
            normalise_a_factors=best_enhanced.normalise_a_factors,
            scale_aware_ties=best_enhanced.scale_aware_ties,
            gpa_module_count=best_enhanced.gpa_module_count,
            gpa_modules_hitting_max_iter=best_enhanced.gpa_modules_hitting_max_iter,
            gpa_max_iter_fraction=best_enhanced.gpa_max_iter_fraction,
        )
        main_rows.append(synthetic_best)

    return {
        "source_root": str(source_root),
        "row_order": [row.method_key for row in main_rows],
        "rows": [serialize_record(row) for row in main_rows],
    }


def build_enhancement_ablation_payload(records: Sequence[SweepRecord], *, source_root: Path) -> Dict[str, object]:
    records_by_method: Dict[str, List[SweepRecord]] = {}
    for record in records:
        if record.method_key.startswith("gpa_"):
            records_by_method.setdefault(record.method_key, []).append(record)

    best_per_variant = {
        method_key: select_best_record(items)
        for method_key, items in records_by_method.items()
        if method_key in GPA_VARIANT_ORDER
    }
    baseline = best_per_variant["gpa_baseline"]
    variants = []
    best_enhanced = None
    enhanced_candidates = []
    for method_key in GPA_VARIANT_ORDER:
        record = best_per_variant.get(method_key)
        if record is None:
            continue
        row = serialize_record(record)
        row["delta_vs_baseline"] = {
            task: float(record.primary_metrics[task]["value"]) - float(baseline.primary_metrics[task]["value"])
            for task in TASK_ORDER
        }
        row["average_delta_vs_baseline"] = record.average_primary_score - baseline.average_primary_score
        variants.append(row)
        if method_key != "gpa_baseline":
            enhanced_candidates.append(record)
    if enhanced_candidates:
        best_enhanced = select_best_record(enhanced_candidates)

    payload = {
        "source_root": str(source_root),
        "baseline": serialize_record(baseline),
        "variants": variants,
    }
    if best_enhanced is not None:
        payload["best_enhanced_variant"] = serialize_record(best_enhanced)
    return payload


def build_gpa_rerun_decision_payload(records: Sequence[SweepRecord], *, source_root: Path) -> Dict[str, object]:
    gpa_records = [record for record in records if record.method_key.startswith("gpa_")]
    records_by_method: Dict[str, List[SweepRecord]] = {}
    for record in gpa_records:
        records_by_method.setdefault(record.method_key, []).append(record)

    best_per_variant = {
        method_key: select_best_record(items)
        for method_key, items in records_by_method.items()
        if method_key in GPA_VARIANT_ORDER
    }
    total_module_count = sum(record.gpa_module_count or 0 for record in gpa_records)
    total_hit_max = sum(record.gpa_modules_hitting_max_iter or 0 for record in gpa_records)
    saturation_fraction = total_hit_max / total_module_count if total_module_count else None

    baseline = best_per_variant.get("gpa_baseline")
    enhanced_candidates = [best_per_variant[key] for key in GPA_VARIANT_ORDER[1:] if key in best_per_variant]
    best_enhanced = select_best_record(enhanced_candidates) if enhanced_candidates else None

    enhanced_delta = None
    if baseline is not None and best_enhanced is not None:
        enhanced_delta = best_enhanced.average_primary_score - baseline.average_primary_score

    gpa_cola_best = max(float(record.primary_metrics["cola"]["value"]) for record in gpa_records) if gpa_records else None
    rerun_recommended = bool(
        (saturation_fraction is not None and saturation_fraction > 0.5)
        and (gpa_cola_best is not None and gpa_cola_best <= 0.0)
        and (enhanced_delta is None or enhanced_delta < 0.01)
    )

    rationale = []
    if saturation_fraction is not None:
        rationale.append(
            f"GPA module saturation is {saturation_fraction:.3f} ({total_hit_max}/{total_module_count} modules hit max_iter)."
        )
    if gpa_cola_best is not None:
        rationale.append(f"Best GPA-family CoLA Matthews is {gpa_cola_best:.4f}.")
    if enhanced_delta is not None and baseline is not None and best_enhanced is not None:
        rationale.append(
            f"Best enhanced GPA average-primary-score delta vs baseline is {enhanced_delta:.4f} "
            f"({best_enhanced.display_name} vs {baseline.display_name})."
        )

    return {
        "source_root": str(source_root),
        "gpa_record_count": len(gpa_records),
        "gpa_module_count": total_module_count,
        "gpa_modules_hitting_max_iter": total_hit_max,
        "gpa_max_iter_saturation_fraction": saturation_fraction,
        "best_per_variant": {
            method_key: serialize_record(record)
            for method_key, record in best_per_variant.items()
        },
        "best_gpa_family_cola_metric": gpa_cola_best,
        "best_enhanced_minus_baseline_avg_primary": enhanced_delta,
        "rerun_recommended": rerun_recommended,
        "candidate_scope": [
            "gpa_baseline_best_trim_region",
            "gpa_dgpa_saties_wb_0p5",
            "gpa_dgpa_saties_wb_1p0",
        ],
        "rationale": rationale,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    sweep_root = Path(args.sweep_root)
    adapters_dir = Path(args.adapters_dir)
    records = load_sweep_records(sweep_root)
    if not records:
        raise ValueError(f"No evaluated sweep records found under {sweep_root}")

    oracle_record = build_oracle_record(adapters_dir)

    best_hparams_payload = build_best_hparams_payload(records, oracle_record, source_root=sweep_root)
    main_results_payload = build_main_results_payload(records, oracle_record, source_root=sweep_root)
    enhancement_ablation_payload = build_enhancement_ablation_payload(records, source_root=sweep_root)
    gpa_rerun_decision_payload = build_gpa_rerun_decision_payload(records, source_root=sweep_root)

    write_json(Path(args.best_hparams_path), best_hparams_payload)
    write_json(Path(args.main_results_path), main_results_payload)
    write_json(Path(args.enhancement_ablation_path), enhancement_ablation_payload)
    write_json(Path(args.gpa_rerun_decision_path), gpa_rerun_decision_payload)

    print(f"Saved best hyperparameters to {args.best_hparams_path}")
    print(f"Saved main results to {args.main_results_path}")
    print(f"Saved enhancement ablation to {args.enhancement_ablation_path}")
    print(f"Saved GPA rerun decision to {args.gpa_rerun_decision_path}")


if __name__ == "__main__":
    main()
