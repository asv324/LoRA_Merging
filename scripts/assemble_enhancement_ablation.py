"""Experiment 9: assemble the restored-head GPA enhancement ablation.

This is a post-hoc reporting script. It consumes the restored-head Week 3
enhancement artifact, extracts the five GPA-family variants at their own best
lambda/trim settings, computes the methodology-declared contribution
decomposition, and renders the headline bar chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover - plotting is optional via --skip-plot
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_task_arithmetic import to_jsonable

TASK_ORDER = ["sst2", "mnli", "qnli", "cola", "rte"]
TASK_LABELS = {
    "sst2": "SST-2",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "cola": "CoLA",
    "rte": "RTE",
}
VARIANT_ORDER = [
    "gpa_baseline",
    "gpa_dgpa_ties",
    "gpa_dgpa_saties",
    "gpa_dgpa_saties_wb_0p5",
    "gpa_dgpa_saties_wb_1p0",
]
VARIANT_LABELS = {
    "gpa_baseline": "GPA+TIES",
    "gpa_dgpa_ties": "dGPA+TIES",
    "gpa_dgpa_saties": "dGPA+saTIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA+saTIES+wB(0.5)",
    "gpa_dgpa_saties_wb_1p0": "dGPA+saTIES+wB(1.0)",
}
VARIANT_COLORS = {
    "gpa_baseline": "#00BCD4",
    "gpa_dgpa_ties": "#5C6BC0",
    "gpa_dgpa_saties": "#FFA726",
    "gpa_dgpa_saties_wb_0p5": "#43A047",
    "gpa_dgpa_saties_wb_1p0": "#EF5350",
}
CONTRIBUTIONS = [
    {
        "effect_key": "directional_alignment",
        "label": "Directional alignment",
        "comparison": "gpa_dgpa_ties",
        "baseline": "gpa_baseline",
        "formula": "dGPA+TIES - GPA+TIES",
    },
    {
        "effect_key": "scale_aware_ties",
        "label": "Scale-aware TIES",
        "comparison": "gpa_dgpa_saties",
        "baseline": "gpa_dgpa_ties",
        "formula": "dGPA+saTIES - dGPA+TIES",
    },
    {
        "effect_key": "b_weight_alpha_0p5",
        "label": "B weighting alpha=0.5",
        "comparison": "gpa_dgpa_saties_wb_0p5",
        "baseline": "gpa_dgpa_saties",
        "formula": "dGPA+saTIES+wB(0.5) - dGPA+saTIES",
    },
    {
        "effect_key": "b_weight_alpha_1p0",
        "label": "B weighting alpha=1.0",
        "comparison": "gpa_dgpa_saties_wb_1p0",
        "baseline": "gpa_dgpa_saties",
        "formula": "dGPA+saTIES+wB(1.0) - dGPA+saTIES",
    },
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        default=PROJECT_ROOT / "results" / "enhancement_ablation_restored_heads.json",
        help="Restored-head enhancement artifact produced by analyze_week3_sweep.py.",
    )
    parser.add_argument("--results-root", default=PROJECT_ROOT / "results" / "ablation_enhancement")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--figure-path",
        default=PROJECT_ROOT / "dissertation" / "chapters" / "figures" / "ablation_enhancement.pdf",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_value(row: Dict[str, object], task: str) -> float:
    return float(row["primary_metrics"][task]["value"])  # type: ignore[index]


def build_variant_table(source_payload: Dict[str, object]) -> List[Dict[str, object]]:
    variants = source_payload.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError("Source payload is missing a 'variants' list")
    lookup = {str(variant["method_key"]): variant for variant in variants if isinstance(variant, dict)}
    missing = [method_key for method_key in VARIANT_ORDER if method_key not in lookup]
    if missing:
        raise ValueError(f"Source payload missing GPA variants: {missing}")

    table = []
    for method_key in VARIANT_ORDER:
        source_row = lookup[method_key]
        hyperparameters = source_row.get("hyperparameters", {})
        table.append(
            {
                "method_key": method_key,
                "display_name": source_row.get("display_name", VARIANT_LABELS[method_key]),
                "source_path": source_row.get("source_path"),
                "hyperparameters": hyperparameters,
                "primary_metrics": source_row["primary_metrics"],
                "average_primary_score": float(source_row["average_primary_score"]),
                "delta_vs_gpa_baseline": {
                    task: float(source_row.get("delta_vs_baseline", {}).get(task, 0.0))  # type: ignore[union-attr]
                    for task in TASK_ORDER
                },
                "average_delta_vs_gpa_baseline": float(source_row.get("average_delta_vs_baseline", 0.0)),
            }
        )
    return table


def delta_row(
    *,
    effect_key: str,
    label: str,
    formula: str,
    comparison: Dict[str, object],
    baseline: Dict[str, object],
) -> Dict[str, object]:
    task_deltas = {
        task: primary_value(comparison, task) - primary_value(baseline, task)
        for task in TASK_ORDER
    }
    return {
        "effect_key": effect_key,
        "label": label,
        "formula": formula,
        "comparison_method_key": comparison["method_key"],
        "baseline_method_key": baseline["method_key"],
        "task_deltas": task_deltas,
        "average_primary_score_delta": float(comparison["average_primary_score"]) - float(baseline["average_primary_score"]),
        "positive_task_count": sum(1 for value in task_deltas.values() if value > 0.0),
    }


def build_contribution_decomposition(variant_table: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    lookup = {row["method_key"]: row for row in variant_table}
    return [
        delta_row(
            effect_key=definition["effect_key"],
            label=definition["label"],
            formula=definition["formula"],
            comparison=lookup[definition["comparison"]],
            baseline=lookup[definition["baseline"]],
        )
        for definition in CONTRIBUTIONS
    ]


def build_summary(source_payload: Dict[str, object], source_path: Path) -> Dict[str, object]:
    variant_table = build_variant_table(source_payload)
    contribution_decomposition = build_contribution_decomposition(variant_table)
    best_variant = max(variant_table, key=lambda row: float(row["average_primary_score"]))
    return {
        "step": "step_4_5_experiment_9_enhancement_ablation",
        "source_path": str(source_path),
        "source_root": source_payload.get("source_root"),
        "task_order": TASK_ORDER,
        "variant_table": variant_table,
        "contribution_decomposition": contribution_decomposition,
        "best_variant": {
            "method_key": best_variant["method_key"],
            "display_name": best_variant["display_name"],
            "average_primary_score": best_variant["average_primary_score"],
        },
    }


def set_publication_style() -> None:
    assert plt is not None
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#37474F",
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "#FAFAFA",
            "savefig.facecolor": "#FAFAFA",
            "legend.frameon": False,
        }
    )


def plot_summary(summary: Dict[str, object], figure_path: Path) -> List[Path]:
    if plt is None or np is None:
        raise SystemExit("matplotlib and numpy are required for plotting. Re-run with --skip-plot to write JSON only.")

    variants = summary["variant_table"]
    decomposition = summary["contribution_decomposition"]

    set_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True, width_ratios=[1.1, 1.0])

    x_values = np.arange(len(variants), dtype=float)
    avg_values = [float(row["average_primary_score"]) for row in variants]
    bars = axes[0].bar(
        x_values,
        avg_values,
        color=[VARIANT_COLORS[row["method_key"]] for row in variants],
        width=0.68,
    )
    for bar, value in zip(bars, avg_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("Enhancement ablation")
    axes[0].set_ylabel("Average primary score")
    axes[0].set_xticks(x_values)
    axes[0].set_xticklabels([VARIANT_LABELS[row["method_key"]] for row in variants], rotation=25, ha="right")
    axes[0].grid(axis="y", linestyle=(0, (4, 4)), alpha=0.7)

    effect_labels = [row["label"] for row in decomposition]
    effect_values = [float(row["average_primary_score_delta"]) for row in decomposition]
    effect_colors = ["#43A047" if value >= 0.0 else "#EF5350" for value in effect_values]
    y_values = np.arange(len(effect_values), dtype=float)
    axes[1].barh(y_values, effect_values, color=effect_colors, height=0.62)
    axes[1].axvline(0.0, color="#37474F", linewidth=1.0)
    for y_pos, value in zip(y_values, effect_values):
        label_x = value + (0.0004 if value >= 0.0 else -0.0004)
        axes[1].text(
            label_x,
            y_pos,
            f"{value:+.4f}",
            va="center",
            ha="left" if value >= 0.0 else "right",
            fontsize=8,
        )
    axes[1].set_title("Contribution decomposition")
    axes[1].set_xlabel("Average score delta")
    axes[1].set_yticks(y_values)
    axes[1].set_yticklabels(effect_labels)
    axes[1].grid(axis="x", linestyle=(0, (4, 4)), alpha=0.7)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = figure_path.with_suffix(".png")
    figure.savefig(figure_path, bbox_inches="tight")
    figure.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
    return [figure_path, png_path]


def main() -> None:
    args = build_arg_parser().parse_args()
    source_path = Path(args.source_path)
    results_root = Path(args.results_root)
    summary_path = Path(args.summary_path) if args.summary_path else results_root / "summary.json"

    source_payload = load_json(source_path)
    summary = build_summary(source_payload, source_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved enhancement ablation summary to {summary_path}")

    print("Contribution decomposition:")
    for row in summary["contribution_decomposition"]:
        print(f"  {row['label']}: {row['average_primary_score_delta']:+.4f} ({row['positive_task_count']}/5 tasks positive)")

    if not args.skip_plot:
        outputs = plot_summary(summary, Path(args.figure_path))
        for path in outputs:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
