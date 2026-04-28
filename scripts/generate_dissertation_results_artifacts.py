"""Generate final dissertation results figures and tables.

The script reads persisted experiment outputs from ``results/`` and writes a
single styled artifact tree under ``dissertation/``. It does not rerun model
training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dissertation_plot_style import (
    COLORS,
    PALETTE,
    SEQUENTIAL_CMAP,
    TASK_LABELS,
    TASK_ORDER,
    add_panel_label,
    apply_style,
    clean_axes,
    figure_size,
    metric,
    relative_paths,
    save_figure,
    write_csv,
    write_latex_table,
)
from scripts.synthetic_exp2_convergence import plot_convergence_figure


RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURE_ROOT = PROJECT_ROOT / "dissertation" / "figures" / "results"
TABLE_ROOT = PROJECT_ROOT / "dissertation" / "tables" / "results"
MANIFEST_PATH = PROJECT_ROOT / "dissertation" / "results_artifacts_manifest.json"

METHOD_LABELS = {
    "oracle": "Individual (oracle)",
    "task_arithmetic": "Task Arithmetic",
    "ties": "TIES",
    "dare_ties": "DARE+TIES",
    "lr_knots": "LR-KnOTS+TIES",
    "gpa_baseline": "GPA+TIES",
    "gpa_best_enhanced": "Best enhanced GPA",
    "gpa_dgpa_ties": "dGPA+TIES",
    "gpa_dgpa_saties": "dGPA+saTIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA+saTIES+wB(0.5)",
    "gpa_dgpa_saties_wb_1p0": "dGPA+saTIES+wB(1.0)",
    "gpa_aligned_ta": "GPA-aligned TA",
    "enhanced_gpa_aligned_ta": "Enhanced-GPA-aligned TA",
}

MAIN_METHOD_ORDER = [
    "oracle",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "lr_knots",
    "gpa_baseline",
    "gpa_best_enhanced",
]

N_ABLATION_METHODS = [
    "gpa_baseline",
    "gpa_dgpa_saties_wb_0p5",
    "ties",
    "lr_knots",
]

METHOD_COLORS = {
    "oracle": COLORS["charcoal"],
    "task_arithmetic": COLORS["amber"],
    "ties": COLORS["slate"],
    "dare_ties": COLORS["coral"],
    "lr_knots": COLORS["blue_grey"],
    "gpa_baseline": COLORS["teal"],
    "gpa_best_enhanced": COLORS["green"],
    "gpa_dgpa_ties": COLORS["slate"],
    "gpa_dgpa_saties": COLORS["amber"],
    "gpa_dgpa_saties_wb_0p5": COLORS["green"],
    "gpa_dgpa_saties_wb_1p0": COLORS["coral"],
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = finite(values)
    return float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0


def get_metric(row: dict[str, Any], task: str) -> float:
    return float(row["primary_metrics"][task]["value"])


def source(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def make_manifest_entry(
    stem: str,
    outputs: list[Path],
    sources: list[Path],
    description: str,
) -> dict[str, Any]:
    return {
        "stem": stem,
        "description": description,
        "outputs": relative_paths(outputs, PROJECT_ROOT),
        "sources": [source(path) for path in sources],
    }


def plot_synthetic_rotation_recovery() -> dict[str, Any]:
    path = RESULTS_ROOT / "synthetic_exp1.json"
    payload = load_json(path)
    summaries = [row for row in payload["config_summaries"] if int(row["rank"]) == 16]
    sigmas = payload["parameters"]["sigmas"]
    adapters = payload["parameters"]["num_adapters"]

    matrix = np.zeros((len(sigmas), len(adapters)), dtype=float)
    for i, sigma in enumerate(sigmas):
        for j, adapter_count in enumerate(adapters):
            match = next(
                row
                for row in summaries
                if float(row["sigma"]) == float(sigma) and int(row["num_adapters"]) == int(adapter_count)
            )
            matrix[i, j] = max(float(match["rotation_recovery_error_mean"]), 1e-8)

    fig, ax = plt.subplots(figsize=figure_size(88, 78))
    image = ax.imshow(np.log10(matrix), cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_title("Rotation recovery at rank 16")
    ax.set_xlabel("Number of adapters")
    ax.set_ylabel("Noise sigma")
    ax.set_xticks(np.arange(len(adapters)), labels=[str(n) for n in adapters])
    ax.set_yticks(np.arange(len(sigmas)), labels=[str(sigma) for sigma in sigmas])
    clean_axes(ax, grid_axis=None)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1e}", ha="center", va="center", fontsize=6, color=COLORS["charcoal"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("log10 rotation error")
    colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])
    outputs = save_figure(fig, FIGURE_ROOT / "synthetic", "fig_04_01_synthetic_rotation_recovery")
    return make_manifest_entry(
        "fig_04_01_synthetic_rotation_recovery",
        outputs,
        [path],
        "Synthetic Experiment 1 rotation recovery heatmap.",
    )


def plot_synthetic_convergence() -> dict[str, Any]:
    path = RESULTS_ROOT / "synthetic_exp2_convergence.json"
    payload = load_json(path)
    outputs = plot_convergence_figure(
        baseline_summary=payload["baseline_summary"],
        sweep_summaries=payload["sweep_summaries"],
        output_dir=FIGURE_ROOT / "synthetic",
        dpi=300,
        output_stem="fig_04_02_synthetic_convergence",
    )
    return make_manifest_entry(
        "fig_04_02_synthetic_convergence",
        outputs,
        [path],
        "Synthetic Experiment 2 convergence curves and iteration summary.",
    )


def plot_synthetic_nonorthogonal() -> dict[str, Any]:
    path = RESULTS_ROOT / "synthetic_exp3_nonorthogonal.json"
    payload = load_json(path)
    rows = payload["config_summaries"]
    deltas = [float(row["delta"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 78), constrained_layout=True)
    axes[0].errorbar(
        deltas,
        [float(row["alignment_residual_mean"]) for row in rows],
        yerr=[float(row["alignment_residual_std"]) for row in rows],
        marker="o",
        color=COLORS["teal"],
    )
    axes[0].set_title("Alignment residual")
    axes[0].set_xlabel("Non-orthogonal perturbation delta")
    axes[0].set_ylabel("Mean residual")
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")

    axes[1].errorbar(
        deltas,
        [float(row["rotation_recovery_error_mean"]) for row in rows],
        yerr=[float(row["rotation_recovery_error_std"]) for row in rows],
        marker="s",
        color=COLORS["coral"],
    )
    axes[1].set_title("Rotation recovery error")
    axes[1].set_xlabel("Non-orthogonal perturbation delta")
    axes[1].set_ylabel("Mean error")
    clean_axes(axes[1])
    add_panel_label(axes[1], "b")

    outputs = save_figure(fig, FIGURE_ROOT / "synthetic", "fig_04_03_synthetic_nonorthogonal_robustness")
    return make_manifest_entry(
        "fig_04_03_synthetic_nonorthogonal_robustness",
        outputs,
        [path],
        "Synthetic Experiment 3 robustness to non-orthogonal perturbations.",
    )


def plot_synthetic_structured() -> dict[str, Any]:
    path = RESULTS_ROOT / "synthetic_exp4_structured.json"
    payload = load_json(path)
    rows = payload["config_summaries"]
    strengths = [float(row["perturbation_strength"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 78), constrained_layout=True)
    axes[0].errorbar(
        strengths,
        [float(row["subspace_overlap_mean"]) for row in rows],
        yerr=[float(row["subspace_overlap_std"]) for row in rows],
        marker="o",
        color=COLORS["teal"],
    )
    axes[0].set_title("Dominant subspace overlap")
    axes[0].set_xlabel("Task-specific perturbation strength")
    axes[0].set_ylabel("Overlap")
    axes[0].set_ylim(0.97, 0.99)
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")

    axes[1].errorbar(
        strengths,
        [float(row["mean_principal_angle_deg_mean"]) for row in rows],
        yerr=[float(row["mean_principal_angle_deg_std"]) for row in rows],
        marker="s",
        color=COLORS["slate"],
    )
    axes[1].set_title("Mean principal angle")
    axes[1].set_xlabel("Task-specific perturbation strength")
    axes[1].set_ylabel("Degrees")
    clean_axes(axes[1])
    add_panel_label(axes[1], "b")

    outputs = save_figure(fig, FIGURE_ROOT / "synthetic", "fig_04_04_synthetic_structured_overlap")
    return make_manifest_entry(
        "fig_04_04_synthetic_structured_overlap",
        outputs,
        [path],
        "Synthetic Experiment 4 structured LoRA-like subspace recovery.",
    )


def layer_index(module_name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def plot_adapter_norm_structure() -> dict[str, Any]:
    path = RESULTS_ROOT / "adapter_norm_analysis.json"
    payload = load_json(path)
    tasks = payload["tasks"]
    labels = [TASK_LABELS[item["task"]] for item in tasks]
    combined_norms = [
        float(item["norm_summary"]["avg_lora_A_norm"]) + float(item["norm_summary"]["avg_lora_B_norm"])
        for item in tasks
    ]

    layers = sorted(
        {
            idx
            for item in tasks
            for idx in [layer_index(layer["module_name"]) for layer in item["layer_norms"]]
            if idx is not None
        }
    )
    heatmap = np.zeros((len(tasks), len(layers)), dtype=float)
    for row_idx, item in enumerate(tasks):
        by_layer: dict[int, list[float]] = defaultdict(list)
        for layer in item["layer_norms"]:
            idx = layer_index(layer["module_name"])
            if idx is not None:
                by_layer[idx].append(float(layer["lora_B_frobenius_norm"]))
        for col_idx, idx in enumerate(layers):
            heatmap[row_idx, col_idx] = mean(by_layer[idx])

    fig, axes = plt.subplots(2, 1, figsize=figure_size(180, 112), constrained_layout=True, height_ratios=[1.0, 1.25])
    order = np.argsort(combined_norms)
    axes[0].barh(
        np.arange(len(tasks)),
        np.array(combined_norms)[order],
        color=[PALETTE[i % len(PALETTE)] for i in range(len(tasks))],
    )
    axes[0].set_yticks(np.arange(len(tasks)), labels=np.array(labels)[order])
    axes[0].set_xlabel("Mean LoRA A + B Frobenius norm")
    axes[0].set_title("Adapter norm ranking")
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")

    image = axes[1].imshow(heatmap, cmap=SEQUENTIAL_CMAP, aspect="auto")
    axes[1].set_title("B-factor norm by task and layer")
    axes[1].set_xlabel("Transformer layer")
    axes[1].set_ylabel("Task")
    axes[1].set_yticks(np.arange(len(labels)), labels=labels)
    axes[1].set_xticks(np.arange(0, len(layers), 3), labels=[str(layers[i]) for i in range(0, len(layers), 3)])
    clean_axes(axes[1], grid_axis=None)
    add_panel_label(axes[1], "b")
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.02, pad=0.02)
    colorbar.set_label("Mean B norm")
    colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])

    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "fig_04_05_adapter_norm_structure")
    return make_manifest_entry(
        "fig_04_05_adapter_norm_structure",
        outputs,
        [path],
        "Experiment 5 adapter norm ranking and layer-local B-factor structure.",
    )


def adapter_sources() -> tuple[Path, Path, dict[str, Any], dict[str, Any], list[str]]:
    analysis_path = RESULTS_ROOT / "adapter_norm_analysis.json"
    mapping_path = PROJECT_ROOT / "configs" / "lora_param_mapping.json"
    analysis = load_json(analysis_path)
    mapping = load_json(mapping_path)
    task_order = list(mapping["tasks_verified"])
    return analysis_path, mapping_path, analysis, mapping, task_order


def task_lookup(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["task"]: item for item in analysis["tasks"]}


def task_color(task: str) -> str:
    return {
        "sst2": COLORS["teal"],
        "mnli": COLORS["slate"],
        "qnli": COLORS["coral"],
        "cola": COLORS["amber"],
        "rte": COLORS["green"],
    }.get(task, COLORS["blue_grey"])


def performance_ratio(check: dict[str, Any]) -> float:
    lower = float(check["target_lower"])
    upper = float(check["target_upper"])
    value = float(check["metric_value"])
    return (value - lower) / (upper - lower) if upper > lower else 0.0


def adapter_manifest_entry(stem: str, outputs: list[Path], sources: list[Path], description: str) -> dict[str, Any]:
    return make_manifest_entry(stem, outputs, sources, f"Experiment 5 adapter analysis: {description}")


def plot_adapter_norm_ranking() -> dict[str, Any]:
    analysis_path, mapping_path, analysis, _, _ = adapter_sources()
    ranking = analysis["norm_ranking"]
    tasks = [row["task"] for row in ranking]
    x = np.arange(len(tasks), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=figure_size(180, 78))
    bars_a = ax.bar(
        x - width / 2,
        [float(row["avg_lora_A_norm"]) for row in ranking],
        width,
        color=COLORS["teal"],
        label="LoRA A",
    )
    bars_b = ax.bar(
        x + width / 2,
        [float(row["avg_lora_B_norm"]) for row in ranking],
        width,
        color=COLORS["slate"],
        label="LoRA B",
    )
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLORS["charcoal"],
            )
    ax.set_title("Per-task adapter norm ranking")
    ax.set_xlabel("Task")
    ax.set_ylabel("Average Frobenius norm")
    ax.set_xticks(x, labels=[TASK_LABELS[task] for task in tasks])
    ax.legend(loc="upper right")
    clean_axes(ax)
    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "adapter_norm_ranking")
    return adapter_manifest_entry(
        "adapter_norm_ranking",
        outputs,
        [analysis_path, mapping_path],
        "grouped LoRA A and B norm ranking by task.",
    )


def plot_adapter_norm_attention_vs_mlp() -> dict[str, Any]:
    analysis_path, mapping_path, analysis, _, task_order = adapter_sources()
    lookup = task_lookup(analysis)
    x = np.arange(len(task_order), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(2, 1, figsize=figure_size(180, 100), constrained_layout=True, sharex=True)
    panels = [
        (
            axes[0],
            "LoRA A norms by module family",
            "avg_attention_lora_A_norm",
            "avg_mlp_lora_A_norm",
            COLORS["teal"],
            COLORS["slate"],
        ),
        (
            axes[1],
            "LoRA B norms by module family",
            "avg_attention_lora_B_norm",
            "avg_mlp_lora_B_norm",
            COLORS["amber"],
            COLORS["coral"],
        ),
    ]
    for panel_index, (ax, title, attention_key, mlp_key, attention_color, mlp_color) in enumerate(panels):
        attention = [float(lookup[task]["norm_summary"][attention_key]) for task in task_order]
        mlp = [float(lookup[task]["norm_summary"][mlp_key]) for task in task_order]
        ax.bar(x - width / 2, attention, width, color=attention_color, label="Attention")
        ax.bar(x + width / 2, mlp, width, color=mlp_color, label="MLP")
        ax.set_title(title)
        ax.set_ylabel("Average Frobenius norm")
        ax.legend(loc="upper right")
        clean_axes(ax)
        add_panel_label(ax, "a" if panel_index == 0 else "b")

    axes[1].set_xlabel("Task")
    axes[1].set_xticks(x, labels=[TASK_LABELS[task] for task in task_order])
    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "adapter_norm_attention_vs_mlp")
    return adapter_manifest_entry(
        "adapter_norm_attention_vs_mlp",
        outputs,
        [analysis_path, mapping_path],
        "module-family split of LoRA A and B norms.",
    )


def build_module_matrix(
    task_order: list[str],
    layers: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    field: str,
) -> tuple[np.ndarray, list[str]]:
    matrix = np.empty((len(layers), len(task_order)), dtype=float)
    labels: list[str] = []
    per_task_layers = {
        task: {entry["module_name"]: entry for entry in lookup[task]["layer_norms"]}
        for task in task_order
    }
    for row_index, layer in enumerate(layers):
        name = str(layer["module_name"])
        labels.append(re.sub(r"^.*layers\.(\d+)\.", r"L\1 ", name).replace("self_attn.", "attn."))
        for column_index, task in enumerate(task_order):
            matrix[row_index, column_index] = float(per_task_layers[task][name][field])
    return matrix, labels


def plot_adapter_layer_heatmap_a() -> dict[str, Any]:
    analysis_path, mapping_path, analysis, mapping, task_order = adapter_sources()
    lookup = task_lookup(analysis)
    module_order = {
        "attention": {"k_proj": 0, "o_proj": 1, "q_proj": 2, "v_proj": 3},
        "mlp": {"down_proj": 0, "gate_proj": 1, "up_proj": 2},
    }

    def sort_key(layer: dict[str, Any]) -> tuple[int, int, str]:
        name = str(layer["module_name"])
        group = str(layer["group"])
        layer_idx = layer_index(name)
        module_type = str(layer.get("module_type", ""))
        return (
            -1 if layer_idx is None else layer_idx,
            module_order.get(group, {}).get(module_type, 99),
            module_type,
        )

    grouped = {
        group: sorted([layer for layer in mapping["layers"] if layer["group"] == group], key=sort_key)
        for group in ("attention", "mlp")
    }
    matrices = []
    labels = []
    for group in ("attention", "mlp"):
        matrix, row_labels = build_module_matrix(task_order, grouped[group], lookup, "lora_A_frobenius_norm")
        matrices.append(matrix)
        labels.append(row_labels)
    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size(180, 165),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [len(labels[0]), len(labels[1])]},
    )
    image = None
    for panel_index, (ax, title, matrix, row_labels, group_name, module_key) in enumerate(
        zip(
            axes,
            ["Attention modules", "MLP modules"],
            matrices,
            labels,
            ["attention", "mlp"],
            ["Rows within each layer: k, o, q, v", "Rows within each layer: down, gate, up"],
        )
    ):
        image = ax.imshow(matrix, aspect="auto", cmap=SEQUENTIAL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}: per-layer LoRA A norm")
        ax.set_xticks(np.arange(len(task_order)), labels=[TASK_LABELS[task] for task in task_order])
        layers = [layer_index(str(layer["module_name"])) for layer in grouped[group_name]]
        layer_values = sorted({layer for layer in layers if layer is not None})
        centers = [
            mean([index for index, layer in enumerate(layers) if layer == layer_value])
            for layer_value in layer_values
        ]
        visible_tick_indices = [index for index, layer in enumerate(layer_values) if layer % 2 == 0]
        ax.set_yticks(
            [centers[index] for index in visible_tick_indices],
            labels=[str(layer_values[index]) for index in visible_tick_indices],
        )
        for layer_value in layer_values[:-1]:
            boundary = max(index for index, layer in enumerate(layers) if layer == layer_value) + 0.5
            ax.axhline(boundary, color=COLORS["background"], linewidth=0.35, alpha=0.8)
        ax.set_ylabel("Transformer layer")
        ax.text(
            0.0,
            -0.08,
            module_key,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=COLORS["charcoal"],
        )
        clean_axes(ax, grid_axis=None)
        add_panel_label(ax, "a" if panel_index == 0 else "b")
    axes[1].set_xlabel("Task")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("LoRA A Frobenius norm")
        colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])
    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "adapter_layer_heatmap_A")
    return adapter_manifest_entry(
        "adapter_layer_heatmap_A",
        outputs,
        [analysis_path, mapping_path],
        "two-panel per-layer LoRA A norm heatmap.",
    )


def plot_adapter_depth_trends() -> dict[str, Any]:
    analysis_path, mapping_path, analysis, _, task_order = adapter_sources()
    lookup = task_lookup(analysis)
    grouped_values: dict[str, dict[str, dict[int, list[float]]]] = {
        "attention": {task: defaultdict(list) for task in task_order},
        "mlp": {task: defaultdict(list) for task in task_order},
    }
    for task in task_order:
        for entry in lookup[task]["layer_norms"]:
            idx = layer_index(str(entry["module_name"]))
            if idx is not None:
                grouped_values[str(entry["group"])][task][idx].append(float(entry["lora_A_frobenius_norm"]))

    fig, axes = plt.subplots(2, 1, figsize=figure_size(180, 100), constrained_layout=True, sharex=True)
    for panel_index, (ax, group) in enumerate(zip(axes, ["attention", "mlp"])):
        for task in task_order:
            layers = sorted(grouped_values[group][task])
            values = [mean(grouped_values[group][task][layer]) for layer in layers]
            ax.plot(layers, values, color=task_color(task), marker="o", label=TASK_LABELS[task])
        ax.set_title(f"Mean LoRA A norm by depth ({group})")
        ax.set_ylabel("Mean Frobenius norm")
        ax.legend(loc="best", ncol=3)
        clean_axes(ax)
        add_panel_label(ax, "a" if panel_index == 0 else "b")
    axes[1].set_xlabel("Transformer layer")
    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "adapter_depth_trends")
    return adapter_manifest_entry(
        "adapter_depth_trends",
        outputs,
        [analysis_path, mapping_path],
        "depth profiles of mean LoRA A norms by task and module family.",
    )


def plot_adapter_perf_vs_norm() -> dict[str, Any]:
    analysis_path, mapping_path, analysis, _, _ = adapter_sources()
    ranking = analysis["norm_ranking"]
    checks = analysis["performance_checks"]

    fig, ax = plt.subplots(figsize=figure_size(88, 78))
    for row in ranking:
        task = row["task"]
        combined = float(row["avg_lora_A_norm"]) + float(row["avg_lora_B_norm"])
        perf = performance_ratio(checks[task])
        ax.scatter(combined, perf, s=44, color=task_color(task), label=TASK_LABELS[task], zorder=3)
        ax.text(combined + 0.04, perf, TASK_LABELS[task], va="center", fontsize=7, color=COLORS["charcoal"])
    ax.axhline(1.0, color=COLORS["reference"], linestyle="--", linewidth=1.0)
    ax.text(6.75, 1.03, "Target upper bound", ha="right", fontsize=7, color=COLORS["charcoal"])
    ax.set_title("Standalone performance versus adapter norm")
    ax.set_xlabel("Combined average LoRA norm (A + B)")
    ax.set_ylabel("Normalised target-band performance")
    clean_axes(ax)
    outputs = save_figure(fig, FIGURE_ROOT / "adapter_analysis", "adapter_perf_vs_norm")
    return adapter_manifest_entry(
        "adapter_perf_vs_norm",
        outputs,
        [analysis_path, mapping_path],
        "scatter of combined adapter norm against standalone performance.",
    )


def plot_main_method_comparison() -> dict[str, Any]:
    path = RESULTS_ROOT / "main_results_restored_heads.json"
    payload = load_json(path)
    rows_by_key = {row["method_key"]: row for row in payload["rows"]}
    rows = [rows_by_key[key] for key in MAIN_METHOD_ORDER if key in rows_by_key]
    labels = [METHOD_LABELS.get(row["method_key"], row["display_name"]) for row in rows]
    values = [float(row["average_primary_score"]) for row in rows]
    task_matrix = np.array([[get_metric(row, task) for task in TASK_ORDER] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 88), constrained_layout=True, width_ratios=[0.9, 1.25])
    bar_colors = [METHOD_COLORS.get(row["method_key"], PALETTE[i % len(PALETTE)]) for i, row in enumerate(rows)]
    axes[0].barh(np.arange(len(rows)), values, color=bar_colors)
    axes[0].set_yticks(np.arange(len(rows)), labels=labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Average primary score")
    axes[0].set_title("Best method by average score")
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")
    for idx, value in enumerate(values):
        axes[0].text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=7, color=COLORS["charcoal"])

    image = axes[1].imshow(task_matrix, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title("Task-level primary metrics")
    axes[1].set_xticks(np.arange(len(TASK_ORDER)), labels=[TASK_LABELS[task] for task in TASK_ORDER])
    axes[1].set_yticks(np.arange(len(rows)), labels=labels)
    clean_axes(axes[1], grid_axis=None)
    add_panel_label(axes[1], "b")
    for i in range(task_matrix.shape[0]):
        for j in range(task_matrix.shape[1]):
            axes[1].text(j, i, f"{task_matrix[i, j]:.2f}", ha="center", va="center", fontsize=6, color=COLORS["charcoal"])
    colorbar = fig.colorbar(image, ax=axes[1], fraction=0.03, pad=0.02)
    colorbar.set_label("Metric value")
    colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])

    outputs = save_figure(fig, FIGURE_ROOT / "main_results", "fig_04_06_main_method_comparison")
    return make_manifest_entry(
        "fig_04_06_main_method_comparison",
        outputs,
        [path],
        "Experiment 6 main method comparison using restored-head best hyperparameters.",
    )


def plot_enhancement_ablation() -> dict[str, Any]:
    path = RESULTS_ROOT / "ablation_enhancement" / "summary.json"
    payload = load_json(path)
    variants = payload["variant_table"]
    contributions = payload["contribution_decomposition"]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 82), constrained_layout=True, width_ratios=[1.0, 1.0])
    variant_labels = [row["display_name"] for row in variants]
    variant_scores = [float(row["average_primary_score"]) for row in variants]
    axes[0].barh(
        np.arange(len(variants)),
        variant_scores,
        color=[METHOD_COLORS.get(row["method_key"], PALETTE[i]) for i, row in enumerate(variants)],
    )
    axes[0].set_yticks(np.arange(len(variants)), labels=variant_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Average primary score")
    axes[0].set_title("GPA-family variants")
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")

    contribution_labels = [row["label"].replace(" alpha=", " a=") for row in contributions]
    contribution_scores = [float(row["average_primary_score_delta"]) for row in contributions]
    colors = [COLORS["teal"] if value >= 0 else COLORS["coral"] for value in contribution_scores]
    axes[1].axvline(0.0, color=COLORS["reference"], linestyle="--", linewidth=1.0)
    axes[1].barh(np.arange(len(contributions)), contribution_scores, color=colors)
    axes[1].set_yticks(np.arange(len(contributions)), labels=contribution_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Delta average score")
    axes[1].set_title("Contribution decomposition")
    clean_axes(axes[1])
    add_panel_label(axes[1], "b")

    outputs = save_figure(fig, FIGURE_ROOT / "ablations", "fig_04_07_enhancement_ablation")
    return make_manifest_entry(
        "fig_04_07_enhancement_ablation",
        outputs,
        [path],
        "Experiment 9 enhancement ablation and contribution decomposition.",
    )


def plot_n_ablation() -> dict[str, Any]:
    path = RESULTS_ROOT / "ablation_N" / "summary.json"
    payload = load_json(path)
    cells = payload["cells"]
    n_values = [2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=figure_size(180, 80))
    markers = ["o", "s", "^", "D"]
    for idx, method_key in enumerate(N_ABLATION_METHODS):
        values = [float(cells[method_key][str(n)]["average_primary_score"]["mean"]) for n in n_values]
        errors = [float(cells[method_key][str(n)]["average_primary_score"]["std"]) for n in n_values]
        ax.errorbar(
            n_values,
            values,
            yerr=errors,
            color=METHOD_COLORS.get(method_key, PALETTE[idx]),
            marker=markers[idx],
            label=METHOD_LABELS.get(method_key, method_key),
        )
    ax.set_title("Performance as the number of merged adapters increases")
    ax.set_xlabel("Number of merged adapters")
    ax.set_ylabel("Average primary score")
    ax.set_xticks(n_values)
    ax.legend(loc="best")
    clean_axes(ax)

    outputs = save_figure(fig, FIGURE_ROOT / "ablations", "fig_04_08_n_ablation")
    return make_manifest_entry(
        "fig_04_08_n_ablation",
        outputs,
        [path],
        "Experiment 11 N-ablation line plot with subset dispersion.",
    )


def plot_cka_before_after() -> dict[str, Any]:
    path = RESULTS_ROOT / "cka" / "summary.json"
    payload = load_json(path)
    before = np.array(payload["average_cka_before_matrix"], dtype=float)
    after = np.array(payload["average_cka_after_matrix"], dtype=float)
    np.fill_diagonal(before, np.nan)
    np.fill_diagonal(after, np.nan)
    values = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    cmap = SEQUENTIAL_CMAP.copy()
    cmap.set_bad(COLORS["light_grey"])

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 76), constrained_layout=True)
    for ax, matrix, title, label in zip(axes, [before, after], ["Before GPA", "After GPA"], ["a", "b"]):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(TASK_ORDER)), labels=[TASK_LABELS[task] for task in TASK_ORDER])
        ax.set_yticks(np.arange(len(TASK_ORDER)), labels=[TASK_LABELS[task] for task in TASK_ORDER])
        clean_axes(ax, grid_axis=None)
        add_panel_label(ax, label)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j] * 1000:.2f}", ha="center", va="center", fontsize=6, color=COLORS["charcoal"])
                else:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=6, color=COLORS["charcoal"])
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("Pairwise CKA x 10^-3")
    colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])

    outputs = save_figure(fig, FIGURE_ROOT / "alignment_analysis", "fig_04_09_cka_before_after")
    return make_manifest_entry(
        "fig_04_09_cka_before_after",
        outputs,
        [path],
        "Experiment 10 pairwise CKA before and after GPA alignment.",
    )


def plot_layer_residual_heatmap() -> dict[str, Any]:
    path = RESULTS_ROOT / "layer_analysis" / "summary.json"
    payload = load_json(path)
    heatmap = payload["heatmap"]
    matrix = np.array(heatmap["matrix"], dtype=float)
    layers = [int(layer) for layer in heatmap["layer_indices"]]
    families = [str(family).replace("_", " ").title() for family in heatmap["families"]]
    correlations = payload["correlations"]

    fig, ax = plt.subplots(figsize=figure_size(88, 100))
    image = ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_title("Layer residual by module family")
    ax.set_xlabel("Module family")
    ax.set_ylabel("Transformer layer")
    ax.set_xticks(np.arange(len(families)), labels=families)
    ax.set_yticks(np.arange(0, len(layers), 2), labels=[str(layers[i]) for i in range(0, len(layers), 2)])
    clean_axes(ax, grid_axis=None)
    annotation = (
        f"Spearman rho: overall {correlations['overall']['rho']:.2f}, "
        f"attention {correlations['attention']['rho']:.2f}, MLP {correlations['mlp']['rho']:.2f}"
    )
    ax.text(0.0, -0.12, annotation, transform=ax.transAxes, fontsize=7, color=COLORS["charcoal"], va="top")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Mean normalised residual")
    colorbar.ax.tick_params(labelsize=7, colors=COLORS["charcoal"])

    outputs = save_figure(fig, FIGURE_ROOT / "alignment_analysis", "fig_04_10_layer_residual_heatmap")
    return make_manifest_entry(
        "fig_04_10_layer_residual_heatmap",
        outputs,
        [path],
        "Experiment 13 per-layer alignment residual heatmap and norm-spread correlations.",
    )


def aggregate_cola_by_lambda(records: list[dict[str, Any]], methods: list[str]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        method_key = str(record["method_key"])
        if method_key in methods and record.get("prediction_distribution_available"):
            grouped[(method_key, float(record["lambda"]))].append(record)

    result: dict[str, list[dict[str, float]]] = {}
    for method_key in methods:
        rows: list[dict[str, float]] = []
        lambdas = sorted({key[1] for key in grouped if key[0] == method_key})
        for lambda_value in lambdas:
            candidates = grouped[(method_key, lambda_value)]
            best = max(candidates, key=lambda item: float(item.get("cola_metric", float("-inf"))))
            counts = best.get("class_counts") or [float("nan"), float("nan")]
            total = float(best.get("total_predictions") or sum(counts))
            class_one_fraction = float(counts[1]) / total if total else float("nan")
            rows.append(
                {
                    "lambda": lambda_value,
                    "cola_metric": float(best["cola_metric"]),
                    "class_one_fraction": class_one_fraction,
                    "dominant_fraction": float(best["dominant_fraction"]),
                }
            )
        result[method_key] = rows
    return result


def plot_cola_prediction_distribution() -> dict[str, Any]:
    path = RESULTS_ROOT / "hp_sweep_restored_heads" / "cola_prediction_distributions.json"
    payload = load_json(path)
    methods = ["task_arithmetic", "ties", "dare_ties", "lr_knots", "gpa_baseline", "gpa_dgpa_saties_wb_0p5"]
    grouped = aggregate_cola_by_lambda(payload["records"], methods)

    fig, axes = plt.subplots(1, 2, figsize=figure_size(180, 78), constrained_layout=True)
    for idx, method_key in enumerate(methods):
        rows = grouped[method_key]
        if not rows:
            continue
        color = METHOD_COLORS.get(method_key, PALETTE[idx % len(PALETTE)])
        axes[0].plot(
            [row["lambda"] for row in rows],
            [row["cola_metric"] for row in rows],
            marker="o",
            color=color,
            label=METHOD_LABELS.get(method_key, method_key),
        )
        axes[1].plot(
            [row["lambda"] for row in rows],
            [row["class_one_fraction"] for row in rows],
            marker="o",
            color=color,
            label=METHOD_LABELS.get(method_key, method_key),
        )

    axes[0].set_title("Best CoLA score across lambda")
    axes[0].set_xlabel("Merge scale lambda")
    axes[0].set_ylabel("CoLA Matthews correlation")
    clean_axes(axes[0])
    add_panel_label(axes[0], "a")

    axes[1].set_title("Predicted class-1 fraction")
    axes[1].set_xlabel("Merge scale lambda")
    axes[1].set_ylabel("Fraction of CoLA predictions")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="best", ncol=1)
    clean_axes(axes[1])
    add_panel_label(axes[1], "b")

    outputs = save_figure(fig, FIGURE_ROOT / "main_results", "fig_04_11_cola_prediction_distribution")
    return make_manifest_entry(
        "fig_04_11_cola_prediction_distribution",
        outputs,
        [path],
        "Experiment 8 CoLA score and prediction distribution across lambda.",
    )


def rows_main_results() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    main_path = RESULTS_ROOT / "main_results_restored_heads.json"
    payload = load_json(main_path)
    rows = []
    for row in payload["rows"]:
        hparams = row.get("hyperparameters", {})
        table_row: dict[str, Any] = {
            "Method": row["display_name"],
            "SST-2": metric(get_metric(row, "sst2")),
            "MNLI": metric(get_metric(row, "mnli")),
            "QNLI": metric(get_metric(row, "qnli")),
            "CoLA": metric(get_metric(row, "cola")),
            "RTE": metric(get_metric(row, "rte")),
            "Avg": metric(float(row["average_primary_score"])),
            "Lambda": metric(hparams.get("lambda"), 2),
            "Trim": "--" if hparams.get("trim_percentage") is None else str(hparams.get("trim_percentage")),
            "Drop": "--" if hparams.get("drop_probability") is None else metric(hparams.get("drop_probability"), 2),
            "B alpha": "--" if hparams.get("b_weight_alpha") is None else metric(hparams.get("b_weight_alpha"), 1),
        }
        rows.append(table_row)
    fields = ["Method", "SST-2", "MNLI", "QNLI", "CoLA", "RTE", "Avg", "Lambda", "Trim", "Drop", "B alpha"]
    return rows, fields, [main_path]


def rows_seed_variance() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    restored_path = RESULTS_ROOT / "seed_variance_restored_heads.json"
    fallback_path = RESULTS_ROOT / "seed_variance.json"
    restored = load_json(restored_path)
    # The restored-head seed file currently only contains the headline enhanced variant.
    # Use the fuller seed-variance file for the comparison table if needed.
    selected_path = restored_path if len(restored.get("methods", {})) > 1 else fallback_path
    payload = load_json(selected_path)
    rows = []
    for method_key, item in payload["methods"].items():
        score = item["average_primary_score"]
        rows.append(
            {
                "Method": item["display_name"],
                "Mean avg": metric(score["mean"]),
                "Std": metric(score["std"]),
                "Seed 42": metric(score["values"][0]),
                "Seed 43": metric(score["values"][1]) if len(score["values"]) > 1 else "--",
                "Seed 44": metric(score["values"][2]) if len(score["values"]) > 2 else "--",
                "Source": source(selected_path),
            }
        )
    fields = ["Method", "Mean avg", "Std", "Seed 42", "Seed 43", "Seed 44", "Source"]
    return rows, fields, [selected_path, restored_path]


def rows_enhancement_ablation() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    path = RESULTS_ROOT / "ablation_enhancement" / "summary.json"
    payload = load_json(path)
    rows = []
    for row in payload["variant_table"]:
        rows.append(
            {
                "Variant": row["display_name"],
                "SST-2": metric(row["primary_metrics"]["sst2"]["value"]),
                "MNLI": metric(row["primary_metrics"]["mnli"]["value"]),
                "QNLI": metric(row["primary_metrics"]["qnli"]["value"]),
                "CoLA": metric(row["primary_metrics"]["cola"]["value"]),
                "RTE": metric(row["primary_metrics"]["rte"]["value"]),
                "Avg": metric(row["average_primary_score"]),
                "Delta vs GPA": metric(row["average_delta_vs_gpa_baseline"]),
            }
        )
    fields = ["Variant", "SST-2", "MNLI", "QNLI", "CoLA", "RTE", "Avg", "Delta vs GPA"]
    return rows, fields, [path]


def rows_enhancement_contributions() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    path = RESULTS_ROOT / "ablation_enhancement" / "summary.json"
    payload = load_json(path)
    rows = []
    for row in payload["contribution_decomposition"]:
        rows.append(
            {
                "Effect": row["label"],
                "Formula": row["formula"],
                "Delta avg": metric(row["average_primary_score_delta"]),
                "Positive tasks": f"{row['positive_task_count']}/5",
            }
        )
    fields = ["Effect", "Formula", "Delta avg", "Positive tasks"]
    return rows, fields, [path]


def rows_n_ablation() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    path = RESULTS_ROOT / "ablation_N" / "summary.json"
    payload = load_json(path)
    cells = payload["cells"]
    rows = []
    for method_key in N_ABLATION_METHODS:
        row = {"Method": METHOD_LABELS.get(method_key, method_key)}
        for n in [2, 3, 4, 5]:
            cell = cells[method_key][str(n)]["average_primary_score"]
            row[f"N={n}"] = f"{metric(cell['mean'])} ({metric(cell['std'])})"
        rows.append(row)
    fields = ["Method", "N=2", "N=3", "N=4", "N=5"]
    return rows, fields, [path]


def rows_ta_aligned() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    path = RESULTS_ROOT / "ablation_ta_aligned" / "summary.json"
    payload = load_json(path)
    source_paths = [path]
    rows = []
    reference = payload["reference"]
    rows.append(
        {
            "Variant": reference["variant_label"],
            "SST-2": metric(reference["per_task_primary"]["sst2"]["value"]),
            "MNLI": metric(reference["per_task_primary"]["mnli"]["value"]),
            "QNLI": metric(reference["per_task_primary"]["qnli"]["value"]),
            "CoLA": metric(reference["per_task_primary"]["cola"]["value"]),
            "RTE": metric(reference["per_task_primary"]["rte"]["value"]),
            "Avg": metric(reference["average_primary_score"]),
            "Delta vs TA": metric(0.0),
        }
    )
    for variant in payload["variants"].values():
        source_paths.append(PROJECT_ROOT / variant["result_path"])
        rows.append(
            {
                "Variant": METHOD_LABELS.get(variant["variant_label"], variant["variant_label"]),
                "SST-2": metric(variant["per_task_primary"]["sst2"]["value"]),
                "MNLI": metric(variant["per_task_primary"]["mnli"]["value"]),
                "QNLI": metric(variant["per_task_primary"]["qnli"]["value"]),
                "CoLA": metric(variant["per_task_primary"]["cola"]["value"]),
                "RTE": metric(variant["per_task_primary"]["rte"]["value"]),
                "Avg": metric(variant["average_primary_score"]),
                "Delta vs TA": metric(variant["delta_vs_task_arithmetic"]["average_primary_score_delta"]),
            }
        )
    fields = ["Variant", "SST-2", "MNLI", "QNLI", "CoLA", "RTE", "Avg", "Delta vs TA"]
    return rows, fields, source_paths


def rows_cka_layer_summary() -> tuple[list[dict[str, Any]], list[str], list[Path]]:
    cka_path = RESULTS_ROOT / "cka" / "summary.json"
    layer_path = RESULTS_ROOT / "layer_analysis" / "summary.json"
    cka = load_json(cka_path)
    layer = load_json(layer_path)
    rows = [
        {
            "Analysis": "Mean pairwise CKA before GPA",
            "Value": metric(cka["average_pairwise_cka_before"], 6),
            "N": str(cka["module_count"]),
        },
        {
            "Analysis": "Mean pairwise CKA after GPA",
            "Value": metric(cka["average_pairwise_cka_after"], 6),
            "N": str(cka["module_count"]),
        },
        {
            "Analysis": "Mean pairwise CKA delta",
            "Value": metric(cka["average_pairwise_cka_delta"], 6),
            "N": str(cka["module_count"]),
        },
        {
            "Analysis": "Residual vs B-norm spread rho (overall)",
            "Value": metric(layer["correlations"]["overall"]["rho"], 3),
            "N": str(layer["correlations"]["overall"]["n"]),
        },
        {
            "Analysis": "Residual vs B-norm spread rho (attention)",
            "Value": metric(layer["correlations"]["attention"]["rho"], 3),
            "N": str(layer["correlations"]["attention"]["n"]),
        },
        {
            "Analysis": "Residual vs B-norm spread rho (MLP)",
            "Value": metric(layer["correlations"]["mlp"]["rho"], 3),
            "N": str(layer["correlations"]["mlp"]["n"]),
        },
    ]
    fields = ["Analysis", "Value", "N"]
    return rows, fields, [cka_path, layer_path]


TABLE_BUILDERS: list[tuple[str, str, str, Callable[[], tuple[list[dict[str, Any]], list[str], list[Path]]]]] = [
    ("table_04_01_main_results", "Main restored-head results by method and task.", "tab:main-restored-results", rows_main_results),
    ("table_04_02_seed_variance", "Three-seed variance summary for primary comparisons.", "tab:seed-variance", rows_seed_variance),
    ("table_04_03_enhancement_ablation", "GPA enhancement ablation by task.", "tab:enhancement-ablation", rows_enhancement_ablation),
    (
        "table_04_04_enhancement_contributions",
        "Pairwise contribution decomposition for GPA enhancements.",
        "tab:enhancement-contributions",
        rows_enhancement_contributions,
    ),
    ("table_04_05_n_ablation", "N-ablation mean average score with subset standard deviation.", "tab:n-ablation", rows_n_ablation),
    ("table_04_06_ta_aligned", "Task Arithmetic in aligned factor space.", "tab:ta-aligned", rows_ta_aligned),
    ("table_04_07_alignment_summary", "CKA and layer residual summary statistics.", "tab:alignment-summary", rows_cka_layer_summary),
]


def generate_tables() -> list[dict[str, Any]]:
    entries = []
    for stem, caption, label, builder in TABLE_BUILDERS:
        rows, fields, sources = builder()
        csv_path = TABLE_ROOT / f"{stem}.csv"
        tex_path = TABLE_ROOT / f"{stem}.tex"
        write_csv(csv_path, rows, fields)
        write_latex_table(tex_path, rows, fields, caption=caption, label=label)
        entries.append(make_manifest_entry(stem, [csv_path, tex_path], sources, caption))
    return entries


FIGURE_BUILDERS: list[Callable[[], dict[str, Any]]] = [
    plot_synthetic_rotation_recovery,
    plot_synthetic_convergence,
    plot_synthetic_nonorthogonal,
    plot_synthetic_structured,
    plot_adapter_norm_structure,
    plot_adapter_norm_ranking,
    plot_adapter_norm_attention_vs_mlp,
    plot_adapter_layer_heatmap_a,
    plot_adapter_depth_trends,
    plot_adapter_perf_vs_norm,
    plot_main_method_comparison,
    plot_enhancement_ablation,
    plot_n_ablation,
    plot_cka_before_after,
    plot_layer_residual_heatmap,
    plot_cola_prediction_distribution,
]


def write_manifest(figures: list[dict[str, Any]], tables: list[dict[str, Any]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": source(Path(__file__)),
        "style_source": "Documentation/dissertation_figure_style_guide.md",
        "scope_source": "Documentation/revised_implementation_plan_v2 (1).md",
        "figure_count": len(figures),
        "table_count": len(tables),
        "figures": figures,
        "tables": tables,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-figures", action="store_true", help="Only write tables and manifest entries.")
    parser.add_argument("--skip-tables", action="store_true", help="Only write figures and manifest entries.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    figures = [] if args.skip_figures else [builder() for builder in FIGURE_BUILDERS]
    tables = [] if args.skip_tables else generate_tables()
    write_manifest(figures, tables)
    print(f"Generated {len(figures)} figure entries and {len(tables)} table entries.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
