"""Generate dissertation-ready figures from adapter validation artifacts.

This script visualizes the outputs of Track B.6:
  - `configs/lora_param_mapping.json`
  - `results/adapter_norm_analysis.json`

It focuses on figures that motivate GPA-based merging by showing that the real
LoRA adapters exhibit structured norm imbalance across tasks, layers, and
module families before any alignment is applied.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = PROJECT_ROOT / "configs" / "lora_param_mapping.json"
DEFAULT_ANALYSIS = PROJECT_ROOT / "results" / "adapter_norm_analysis.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")
TASK_COLORS = {
    "sst2": "#4C78A8",
    "mnli": "#F58518",
    "qnli": "#54A24B",
    "cola": "#E45756",
    "rte": "#B279A2",
}


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_layer_index(module_name: str) -> int:
    match = LAYER_INDEX_PATTERN.search(module_name)
    if match is None:
        raise ValueError(f"Could not parse layer index from module name: {module_name}")
    return int(match.group(1))


def build_layer_label(module_name: str) -> str:
    layer_index = parse_layer_index(module_name)
    suffix = re.sub(r"^.*?\.layers\.\d+\.", "", module_name)
    suffix = suffix.replace("self_attn.", "attn.")
    return f"L{layer_index:02d} {suffix}"


def set_publication_style() -> None:
    require_matplotlib()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def require_matplotlib() -> None:
    if plt is None:
        raise SystemExit(
            "matplotlib is required to generate adapter-analysis figures. Install it with `pip install matplotlib`."
        ) from MATPLOTLIB_IMPORT_ERROR


def build_task_lookup(task_summaries: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {summary["task"]: summary for summary in task_summaries}


def canonical_layers_by_group(mapping_payload: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {"attention": [], "mlp": []}
    for entry in mapping_payload["layers"]:
        grouped[entry["group"]].append(entry)
    return grouped


def layer_norm_lookup(task_summary: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {entry["module_name"]: entry for entry in task_summary["layer_norms"]}


def build_group_matrix(
    task_order: Sequence[str],
    group_layers: Sequence[Dict[str, object]],
    task_lookup: Dict[str, Dict[str, object]],
    field_name: str,
) -> Tuple[np.ndarray, List[str]]:
    matrix = np.empty((len(group_layers), len(task_order)), dtype=float)
    labels: List[str] = []

    per_task_lookups = {
        task: layer_norm_lookup(task_lookup[task])
        for task in task_order
    }

    for row_index, layer_entry in enumerate(group_layers):
        module_name = layer_entry["module_name"]
        labels.append(build_layer_label(module_name))
        for col_index, task in enumerate(task_order):
            matrix[row_index, col_index] = float(per_task_lookups[task][module_name][field_name])

    return matrix, labels


def annotate_bar_values(ax: plt.Axes, bars: Iterable[plt.Rectangle], decimals: int = 2) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def performance_ratio(performance_check: Dict[str, object]) -> float:
    lower = float(performance_check["target_lower"])
    upper = float(performance_check["target_upper"])
    value = float(performance_check["metric_value"])
    denominator = upper - lower
    if denominator <= 0:
        return 0.0
    return (value - lower) / denominator


def plot_norm_ranking(
    norm_ranking: Sequence[Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    tasks = [entry["task"] for entry in norm_ranking]
    a_values = [float(entry["avg_lora_A_norm"]) for entry in norm_ranking]
    b_values = [float(entry["avg_lora_B_norm"]) for entry in norm_ranking]

    x = np.arange(len(tasks), dtype=float)
    width = 0.36

    figure, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    bars_a = ax.bar(x - width / 2, a_values, width, label="Average LoRA A norm", color="#4C78A8")
    bars_b = ax.bar(x + width / 2, b_values, width, label="Average LoRA B norm", color="#F58518")

    annotate_bar_values(ax, bars_a)
    annotate_bar_values(ax, bars_b)
    ax.set_title("Per-task adapter norm ranking")
    ax.set_xlabel("Task")
    ax.set_ylabel("Average Frobenius norm")
    ax.set_xticks(x)
    ax.set_xticklabels([task.upper() for task in tasks])
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    pdf_path = output_dir / "adapter_norm_ranking.pdf"
    png_path = output_dir / "adapter_norm_ranking.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_attention_vs_mlp(
    task_order: Sequence[str],
    task_lookup: Dict[str, Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    x = np.arange(len(task_order), dtype=float)
    width = 0.34

    a_attention = [float(task_lookup[task]["norm_summary"]["avg_attention_lora_A_norm"]) for task in task_order]
    a_mlp = [float(task_lookup[task]["norm_summary"]["avg_mlp_lora_A_norm"]) for task in task_order]
    b_attention = [float(task_lookup[task]["norm_summary"]["avg_attention_lora_B_norm"]) for task in task_order]
    b_mlp = [float(task_lookup[task]["norm_summary"]["avg_mlp_lora_B_norm"]) for task in task_order]

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)

    bars_a_attention = axes[0].bar(x - width / 2, a_attention, width, label="Attention", color="#4C78A8")
    bars_a_mlp = axes[0].bar(x + width / 2, a_mlp, width, label="MLP", color="#72B7B2")
    axes[0].set_title("LoRA A norms by module family")
    axes[0].set_ylabel("Average Frobenius norm")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bar_values(axes[0], bars_a_attention)
    annotate_bar_values(axes[0], bars_a_mlp)

    bars_b_attention = axes[1].bar(x - width / 2, b_attention, width, label="Attention", color="#F58518")
    bars_b_mlp = axes[1].bar(x + width / 2, b_mlp, width, label="MLP", color="#E45756")
    axes[1].set_title("LoRA B norms by module family")
    axes[1].set_ylabel("Average Frobenius norm")
    axes[1].set_xlabel("Task")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    annotate_bar_values(axes[1], bars_b_attention)
    annotate_bar_values(axes[1], bars_b_mlp)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([task.upper() for task in task_order])

    pdf_path = output_dir / "adapter_norm_attention_vs_mlp.pdf"
    png_path = output_dir / "adapter_norm_attention_vs_mlp.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_layer_heatmap(
    task_order: Sequence[str],
    grouped_layers: Dict[str, List[Dict[str, object]]],
    task_lookup: Dict[str, Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    matrices = []
    labels = []
    for group_name in ("attention", "mlp"):
        matrix, group_labels = build_group_matrix(
            task_order=task_order,
            group_layers=grouped_layers[group_name],
            task_lookup=task_lookup,
            field_name="lora_A_frobenius_norm",
        )
        matrices.append(matrix)
        labels.append(group_labels)

    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)

    height = 7 + (len(labels[0]) + len(labels[1])) * 0.10
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, max(16, height)),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [len(labels[0]), len(labels[1])]},
    )

    image = None
    for ax, title, matrix, row_labels in zip(
        axes,
        ("Attention modules: per-layer LoRA A norm", "MLP modules: per-layer LoRA A norm"),
        matrices,
        labels,
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(task_order)))
        ax.set_xticklabels([task.upper() for task in task_order])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=5)
        ax.set_ylabel("Canonical module order")

    axes[1].set_xlabel("Task")

    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.92)
        colorbar.set_label("LoRA A Frobenius norm")

    pdf_path = output_dir / "adapter_layer_heatmap_A.pdf"
    png_path = output_dir / "adapter_layer_heatmap_A.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def aggregate_depth_series(
    task_lookup: Dict[str, Dict[str, object]],
    task_order: Sequence[str],
) -> Dict[str, Dict[str, List[float]]]:
    aggregated = {
        "attention": {task: [] for task in task_order},
        "mlp": {task: [] for task in task_order},
    }

    for task in task_order:
        grouped_values: Dict[str, Dict[int, List[float]]] = {
            "attention": defaultdict(list),
            "mlp": defaultdict(list),
        }
        for entry in task_lookup[task]["layer_norms"]:
            layer_index = parse_layer_index(entry["module_name"])
            grouped_values[entry["group"]][layer_index].append(float(entry["lora_A_frobenius_norm"]))

        for group_name in ("attention", "mlp"):
            max_layer = max(grouped_values[group_name])
            aggregated[group_name][task] = [
                float(np.mean(grouped_values[group_name][layer_index]))
                for layer_index in range(max_layer + 1)
            ]

    return aggregated


def plot_depth_trends(
    task_order: Sequence[str],
    task_lookup: Dict[str, Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    aggregated = aggregate_depth_series(task_lookup=task_lookup, task_order=task_order)
    layer_range = np.arange(len(next(iter(aggregated["attention"].values()))))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    for axis, group_name in zip(axes, ("attention", "mlp")):
        for task in task_order:
            axis.plot(
                layer_range,
                aggregated[group_name][task],
                label=task.upper(),
                linewidth=2,
                color=TASK_COLORS.get(task),
            )
        axis.set_title(f"Mean LoRA A norm by depth ({group_name})")
        axis.set_ylabel("Mean Frobenius norm")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[1].set_xlabel("Transformer layer")

    pdf_path = output_dir / "adapter_depth_trends.pdf"
    png_path = output_dir / "adapter_depth_trends.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def plot_performance_vs_norm(
    norm_ranking: Sequence[Dict[str, object]],
    performance_checks: Dict[str, Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    combined_norms = {
        entry["task"]: float(entry["avg_lora_A_norm"]) + float(entry["avg_lora_B_norm"])
        for entry in norm_ranking
    }

    figure, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for task, combined_norm in combined_norms.items():
        y_value = performance_ratio(performance_checks[task])
        ax.scatter(
            combined_norm,
            y_value,
            s=90,
            color=TASK_COLORS.get(task),
            label=task.upper(),
        )
        ax.text(combined_norm + 0.03, y_value, task.upper(), va="center", fontsize=9)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title("Standalone performance relative to expected target range")
    ax.set_xlabel("Combined average LoRA norm (A + B)")
    ax.set_ylabel("Normalized performance within target band")
    ax.grid(alpha=0.25)

    pdf_path = output_dir / "adapter_perf_vs_norm.pdf"
    png_path = output_dir / "adapter_perf_vs_norm.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [pdf_path, png_path]


def write_manifest(
    output_dir: Path,
    mapping_path: Path,
    analysis_path: Path,
    generated_paths: Sequence[Path],
    task_order: Sequence[str],
    mapping_payload: Dict[str, object],
) -> Path:
    manifest_path = output_dir / "adapter_analysis_manifest.json"
    manifest_payload = {
        "source_files": {
            "mapping": str(mapping_path),
            "analysis": str(analysis_path),
        },
        "tasks": list(task_order),
        "verification_summary": mapping_payload["verification_summary"],
        "figure_descriptions": {
            "adapter_norm_ranking": "Grouped bars comparing average LoRA A and LoRA B norms per task.",
            "adapter_norm_attention_vs_mlp": "Module-family split showing how attention and MLP norms differ by task.",
            "adapter_layer_heatmap_A": "Two-panel heatmap of per-layer LoRA A norms across tasks for attention and MLP modules.",
            "adapter_depth_trends": "Depth profiles of mean LoRA A norms by task for attention and MLP families.",
            "adapter_perf_vs_norm": "Exploratory scatter comparing combined norm magnitude with normalized standalone performance.",
        },
        "generated_files": [str(path) for path in generated_paths],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-input", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--analysis-input", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_publication_style()

    mapping_payload = load_json(args.mapping_input)
    norm_payload = load_json(args.analysis_input)
    task_order = list(mapping_payload["tasks_verified"])
    task_lookup = build_task_lookup(norm_payload["tasks"])
    grouped_layers = canonical_layers_by_group(mapping_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []

    generated_paths.extend(plot_norm_ranking(norm_payload["norm_ranking"], output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(plot_attention_vs_mlp(task_order, task_lookup, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(plot_layer_heatmap(task_order, grouped_layers, task_lookup, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(plot_depth_trends(task_order, task_lookup, output_dir=args.output_dir, dpi=args.dpi))
    generated_paths.extend(
        plot_performance_vs_norm(
            norm_payload["norm_ranking"],
            norm_payload["performance_checks"],
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    )

    manifest_path = write_manifest(
        output_dir=args.output_dir,
        mapping_path=args.mapping_input,
        analysis_path=args.analysis_input,
        generated_paths=generated_paths,
        task_order=task_order,
        mapping_payload=mapping_payload,
    )
    generated_paths.append(manifest_path)

    for path in generated_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
