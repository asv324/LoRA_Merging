"""Generate dissertation-ready Week 3 results figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "matplotlib is required to generate Week 3 results figures. Install it with `pip install matplotlib`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_week3_sweep import METHOD_LABELS, load_sweep_records


TASK_ORDER = ["sst2", "mnli", "qnli", "cola", "rte"]
TASK_LABELS = {
    "sst2": "SST-2",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "cola": "CoLA",
    "rte": "RTE",
}
MAIN_METHOD_ORDER = [
    "task_arithmetic",
    "ties",
    "dare_ties",
    "lr_knots",
    "gpa_baseline",
    "gpa_best_enhanced",
]
MAIN_METHOD_LABELS = {
    "task_arithmetic": "Task Arithmetic",
    "ties": "TIES",
    "dare_ties": "DARE+TIES",
    "lr_knots": "LR-KnOTS+TIES",
    "gpa_baseline": "GPA+TIES",
    "gpa_best_enhanced": "Best enhanced GPA",
}
MAIN_METHOD_COLORS = {
    "task_arithmetic": "#FFA726",
    "ties": "#5C6BC0",
    "dare_ties": "#EF5350",
    "lr_knots": "#78909C",
    "gpa_baseline": "#00BCD4",
    "gpa_best_enhanced": "#43A047",
}
GPA_VARIANT_ORDER = [
    "gpa_baseline",
    "gpa_dgpa_ties",
    "gpa_dgpa_saties",
    "gpa_dgpa_saties_wb_0p5",
    "gpa_dgpa_saties_wb_1p0",
]
GPA_VARIANT_LABELS = {
    "gpa_baseline": "GPA+TIES",
    "gpa_dgpa_ties": "dGPA+TIES",
    "gpa_dgpa_saties": "dGPA+saTIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA+saTIES+wB(0.5)",
    "gpa_dgpa_saties_wb_1p0": "dGPA+saTIES+wB(1.0)",
}
GPA_VARIANT_COLORS = {
    "gpa_baseline": "#00BCD4",
    "gpa_dgpa_ties": "#5C6BC0",
    "gpa_dgpa_saties": "#FFA726",
    "gpa_dgpa_saties_wb_0p5": "#43A047",
    "gpa_dgpa_saties_wb_1p0": "#EF5350",
}
CLASS_COLORS = ["#5C6BC0", "#EF5350"]
OUTPUT_BASENAMES = [
    "fig_results_main_methods",
    "fig_results_gpa_ablation",
    "fig_results_lambda_sweeps",
    "fig_results_gpa_rerun",
]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#37474F",
            "axes.linewidth": 1.0,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "#FAFAFA",
            "savefig.facecolor": "#FAFAFA",
            "grid.color": "#CFD8DC",
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
            "legend.frameon": False,
        }
    )


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def abbreviate_method(method_key: str) -> str:
    return MAIN_METHOD_LABELS.get(method_key, METHOD_LABELS.get(method_key, method_key))


def build_main_method_payload(main_results_payload: Dict[str, object]) -> List[Dict[str, object]]:
    row_lookup = {row["method_key"]: row for row in main_results_payload["rows"]}
    best_enhanced = row_lookup["gpa_best_enhanced"]
    row_lookup["gpa_best_enhanced"] = best_enhanced
    return [row_lookup[method_key] for method_key in MAIN_METHOD_ORDER]


def plot_main_methods(
    main_results_payload: Dict[str, object],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    methods = build_main_method_payload(main_results_payload)
    oracle = next(row for row in main_results_payload["rows"] if row["method_key"] == "oracle")

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True, height_ratios=[3.1, 1.8])

    x = np.arange(len(TASK_ORDER), dtype=float)
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(methods))

    for offset, row in zip(offsets, methods):
        method_key = row["method_key"]
        values = [float(row["primary_metrics"][task]["value"]) for task in TASK_ORDER]
        axes[0].bar(
            x + offset,
            values,
            width,
            color=MAIN_METHOD_COLORS[method_key],
            label=abbreviate_method(method_key),
        )

    axes[0].set_title("Best hyperparameter results by task")
    axes[0].set_ylabel("Primary validation metric")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER])
    axes[0].grid(axis="y", linestyle=(0, (4, 4)))
    axes[0].legend(ncol=3, loc="upper center")

    avg_x = np.arange(len(methods), dtype=float)
    avg_values = [float(row["average_primary_score"]) for row in methods]
    avg_bars = axes[1].bar(
        avg_x,
        avg_values,
        color=[MAIN_METHOD_COLORS[row["method_key"]] for row in methods],
        width=0.65,
    )
    oracle_avg = float(oracle["average_primary_score"])
    axes[1].axhline(oracle_avg, color="#37474F", linestyle="--", linewidth=1.2)
    axes[1].text(
        len(methods) - 0.5,
        oracle_avg + 0.005,
        f"Oracle avg = {oracle_avg:.3f}",
        ha="right",
        va="bottom",
        color="#37474F",
        fontsize=8,
    )
    for bar, value in zip(avg_bars, avg_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.003,
            format_metric(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axes[1].set_title("Average primary score")
    axes[1].set_ylabel("Average over 5 tasks")
    axes[1].set_xticks(avg_x)
    axes[1].set_xticklabels([abbreviate_method(row["method_key"]) for row in methods], rotation=20, ha="right")
    axes[1].grid(axis="y", linestyle=(0, (4, 4)))

    return save_figure(figure, output_dir=output_dir, stem="fig_results_main_methods", dpi=dpi)


def plot_gpa_ablation(
    enhancement_payload: Dict[str, object],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    variants = enhancement_payload["variants"]
    variant_lookup = {variant["method_key"]: variant for variant in variants}
    ordered_variants = [variant_lookup[method_key] for method_key in GPA_VARIANT_ORDER]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True, width_ratios=[1.0, 1.45])

    avg_x = np.arange(len(ordered_variants), dtype=float)
    avg_values = [float(variant["average_primary_score"]) for variant in ordered_variants]
    avg_bars = axes[0].bar(
        avg_x,
        avg_values,
        color=[GPA_VARIANT_COLORS[variant["method_key"]] for variant in ordered_variants],
        width=0.68,
    )
    for bar, value in zip(avg_bars, avg_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.0025,
            format_metric(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[0].set_title("GPA-family ablation: average score")
    axes[0].set_ylabel("Average primary score")
    axes[0].set_xticks(avg_x)
    axes[0].set_xticklabels([GPA_VARIANT_LABELS[variant["method_key"]] for variant in ordered_variants], rotation=25, ha="right")
    axes[0].grid(axis="y", linestyle=(0, (4, 4)))

    delta_matrix = np.array(
        [
            [float(variant["delta_vs_baseline"][task]) for task in TASK_ORDER]
            for variant in ordered_variants
        ],
        dtype=float,
    )
    vmax = max(0.001, float(np.max(np.abs(delta_matrix))))
    image = axes[1].imshow(delta_matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Per-task delta vs GPA+TIES baseline")
    axes[1].set_xticks(range(len(TASK_ORDER)))
    axes[1].set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER])
    axes[1].set_yticks(range(len(ordered_variants)))
    axes[1].set_yticklabels([GPA_VARIANT_LABELS[variant["method_key"]] for variant in ordered_variants])
    for row_index in range(delta_matrix.shape[0]):
        for col_index in range(delta_matrix.shape[1]):
            axes[1].text(
                col_index,
                row_index,
                f"{delta_matrix[row_index, col_index]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#37474F",
            )
    colorbar = figure.colorbar(image, ax=axes[1], shrink=0.92)
    colorbar.set_label("Metric delta")

    return save_figure(figure, output_dir=output_dir, stem="fig_results_gpa_ablation", dpi=dpi)


def select_best_projection(records: Sequence[object], method_key: str) -> Dict[float, float]:
    projected: Dict[float, float] = {}
    filtered = [record for record in records if record.method_key == method_key]
    for record in filtered:
        lambda_value = float(record.lambda_value)
        current = projected.get(lambda_value)
        if current is None or record.average_primary_score > current:
            projected[lambda_value] = float(record.average_primary_score)
    return dict(sorted(projected.items()))


def select_best_enhanced_projection(records: Sequence[object]) -> Dict[float, float]:
    projected: Dict[float, float] = {}
    for record in records:
        if record.method_key not in {
            "gpa_dgpa_ties",
            "gpa_dgpa_saties",
            "gpa_dgpa_saties_wb_0p5",
            "gpa_dgpa_saties_wb_1p0",
        }:
            continue
        lambda_value = float(record.lambda_value)
        current = projected.get(lambda_value)
        if current is None or record.average_primary_score > current:
            projected[lambda_value] = float(record.average_primary_score)
    return dict(sorted(projected.items()))


def plot_lambda_sweeps(
    sweep_records: Sequence[object],
    main_results_payload: Dict[str, object],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    figure, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    sweep_methods = [
        ("task_arithmetic", "Task Arithmetic"),
        ("ties", "TIES"),
        ("dare_ties", "DARE+TIES"),
        ("lr_knots", "LR-KnOTS+TIES"),
        ("gpa_baseline", "GPA+TIES"),
    ]

    for method_key, label in sweep_methods:
        projection = select_best_projection(sweep_records, method_key)
        x_values = list(projection.keys())
        y_values = list(projection.values())
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0 if method_key.startswith("gpa") else 1.6,
            markersize=5,
            color=MAIN_METHOD_COLORS.get(method_key, "#78909C"),
            label=label,
        )

    enhanced_projection = select_best_enhanced_projection(sweep_records)
    ax.plot(
        list(enhanced_projection.keys()),
        list(enhanced_projection.values()),
        marker="D",
        linewidth=2.0,
        markersize=5,
        color=MAIN_METHOD_COLORS["gpa_best_enhanced"],
        label="Best enhanced GPA per λ",
    )

    oracle = next(row for row in main_results_payload["rows"] if row["method_key"] == "oracle")
    oracle_avg = float(oracle["average_primary_score"])
    ax.axhline(oracle_avg, color="#37474F", linestyle="--", linewidth=1.0)
    ax.text(1.0, oracle_avg + 0.004, "Oracle avg", ha="right", va="bottom", fontsize=8, color="#37474F")

    ax.set_title("Lambda sweeps using best score at each λ")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Average primary score")
    ax.grid(True, linestyle=(0, (4, 4)))
    ax.legend(ncol=3, loc="lower right")
    ax.set_xlim(0.04, 1.02)

    return save_figure(figure, output_dir=output_dir, stem="fig_results_lambda_sweeps", dpi=dpi)


def format_rerun_label(method_key: str) -> str:
    mapping = {
        "gpa_baseline": "GPA+TIES",
        "gpa_dgpa_saties_wb_0p5": "wB(0.5)",
        "gpa_dgpa_saties_wb_1p0": "wB(1.0)",
    }
    return mapping.get(method_key, method_key)


def plot_gpa_rerun(
    rerun_payload: Dict[str, object],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    results = rerun_payload["rerun_results"]
    method_keys = list(results.keys())
    labels = [format_rerun_label(method_key) for method_key in method_keys]
    x = np.arange(len(method_keys), dtype=float)
    width = 0.34

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)

    saturation_before = [float(results[key]["gpa_max_iter_fraction_before"]) for key in method_keys]
    saturation_after = [float(results[key]["gpa_max_iter_fraction_after"]) for key in method_keys]
    axes[0].bar(x - width / 2, saturation_before, width, color="#EF5350", label="Before rerun")
    axes[0].bar(x + width / 2, saturation_after, width, color="#43A047", label="After rerun")
    axes[0].set_title("GPA saturation fraction")
    axes[0].set_ylabel("Modules hitting max_iter")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", linestyle=(0, (4, 4)))
    axes[0].legend(loc="upper right")

    avg_before = [float(results[key]["average_primary_score_before"]) for key in method_keys]
    avg_after = [float(results[key]["average_primary_score_after"]) for key in method_keys]
    axes[1].bar(x - width / 2, avg_before, width, color="#5C6BC0", label="Before rerun")
    axes[1].bar(x + width / 2, avg_after, width, color="#00BCD4", label="After rerun")
    for x_value, before, after in zip(x, avg_before, avg_after):
        delta = after - before
        axes[1].text(
            x_value,
            max(before, after) + 0.003,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#37474F",
        )
    axes[1].set_title("Average primary score")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis="y", linestyle=(0, (4, 4)))

    class_zero = [int(results[key]["cola_prediction_distribution"]["class_counts"][0]) for key in method_keys]
    class_one = [int(results[key]["cola_prediction_distribution"]["class_counts"][1]) for key in method_keys]
    axes[2].bar(x, class_zero, width=0.6, color=CLASS_COLORS[0], label="Predicted class 0")
    axes[2].bar(x, class_one, width=0.6, bottom=class_zero, color=CLASS_COLORS[1], label="Predicted class 1")
    for x_value, count in zip(x, class_zero):
        axes[2].text(x_value, count + 15, str(count), ha="center", va="bottom", fontsize=8)
    axes[2].set_title("CoLA prediction distribution after rerun")
    axes[2].set_ylabel("Validation examples")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].grid(axis="y", linestyle=(0, (4, 4)))
    axes[2].legend(loc="upper right")

    return save_figure(figure, output_dir=output_dir, stem="fig_results_gpa_rerun", dpi=dpi)


def write_manifest(output_dir: Path, generated_paths: Sequence[Path]) -> Path:
    manifest_payload = {
        "generated_files": [str(path) for path in generated_paths],
        "descriptions": {
            "fig_results_main_methods": "Best-hyperparameter comparison across primary Week 3 methods and average score.",
            "fig_results_gpa_ablation": "GPA variant averages with per-task deltas relative to GPA+TIES baseline.",
            "fig_results_lambda_sweeps": "Lambda sweep trends using the best available score at each lambda value.",
            "fig_results_gpa_rerun": "Targeted rerun diagnostic comparing convergence, average score, and CoLA prediction collapse.",
        },
        "lambda_projection_rule": "For each method family and lambda, select the highest average_primary_score across the remaining sweep hyperparameters. For enhanced GPA, the line uses the best enhanced variant at each lambda.",
    }
    manifest_path = output_dir / "week3_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-results",
        type=Path,
        default=PROJECT_ROOT / "results" / "main_results.json",
    )
    parser.add_argument(
        "--enhancement-ablation",
        type=Path,
        default=PROJECT_ROOT / "results" / "enhancement_ablation.json",
    )
    parser.add_argument(
        "--gpa-rerun-decision",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpa_rerun_decision.json",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "hp_sweep_low_storage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "week3",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_publication_style()

    main_results_payload = load_json(args.main_results)
    enhancement_payload = load_json(args.enhancement_ablation)
    rerun_payload = load_json(args.gpa_rerun_decision)
    sweep_records = load_sweep_records(args.sweep_root)

    generated_paths: List[Path] = []
    generated_paths.extend(plot_main_methods(main_results_payload, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(plot_gpa_ablation(enhancement_payload, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(
        plot_lambda_sweeps(sweep_records, main_results_payload=main_results_payload, output_dir=args.output_dir, dpi=args.dpi)
    )
    generated_paths.extend(plot_gpa_rerun(rerun_payload, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.append(write_manifest(args.output_dir, generated_paths))

    for path in generated_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
