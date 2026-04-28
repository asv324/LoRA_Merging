"""Plot dissertation-ready heatmaps for Synthetic Experiment 1.

Reads `results/synthetic_exp1.json` and renders heatmaps of the selected summary
metric versus noise level (sigma) and number of adapters (N), with one panel
per rank value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "matplotlib is required to plot heatmaps. Install it with `pip install matplotlib`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "synthetic_exp1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

METRIC_LABELS = {
    "rotation_recovery_error_mean": "Mean Rotation Recovery Error",
    "consensus_relative_error_mean": "Mean Consensus Relative Error",
    "alignment_residual_mean": "Mean Alignment Residual",
}


def load_results(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_axes(payload: Dict[str, object]) -> Tuple[List[float], List[int], List[int]]:
    parameters = payload["parameters"]
    return (
        [float(value) for value in parameters["sigmas"]],
        [int(value) for value in parameters["num_adapters"]],
        [int(value) for value in parameters["ranks"]],
    )


def build_heatmap_matrix(
    summaries: Sequence[Dict[str, object]],
    rank: int,
    sigmas: Sequence[float],
    num_adapters_values: Sequence[int],
    metric: str,
) -> np.ndarray:
    lookup = {
        (float(summary["sigma"]), int(summary["num_adapters"]), int(summary["rank"])): float(summary[metric])
        for summary in summaries
    }

    matrix = np.empty((len(num_adapters_values), len(sigmas)), dtype=float)
    for row_index, num_adapters in enumerate(num_adapters_values):
        for col_index, sigma in enumerate(sigmas):
            key = (float(sigma), int(num_adapters), int(rank))
            if key not in lookup:
                raise ValueError(f"missing summary for sigma={sigma}, N={num_adapters}, r={rank}")
            matrix[row_index, col_index] = lookup[key]

    return matrix


def annotate_cells(ax: plt.Axes, matrix: np.ndarray) -> None:
    max_value = float(matrix.max()) if matrix.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = "white" if value > threshold else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2e}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )


def plot_heatmaps(
    payload: Dict[str, object],
    metric: str,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> List[Path]:
    sigmas, num_adapters_values, ranks = extract_axes(payload)
    summaries = payload["config_summaries"]

    matrices = [
        build_heatmap_matrix(summaries, rank, sigmas, num_adapters_values, metric)
        for rank in ranks
    ]

    vmin = min(float(matrix.min()) for matrix in matrices)
    vmax = max(float(matrix.max()) for matrix in matrices)
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()

    image = None
    for ax, rank, matrix in zip(axes, ranks, matrices):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        annotate_cells(ax, matrix)
        ax.set_title(f"Rank r = {rank}")
        ax.set_xlabel("Noise sigma")
        ax.set_ylabel("Number of adapters N")
        ax.set_xticks(range(len(sigmas)))
        ax.set_xticklabels([f"{sigma:g}" for sigma in sigmas])
        ax.set_yticks(range(len(num_adapters_values)))
        ax.set_yticklabels([str(value) for value in num_adapters_values])

    if image is not None:
        colorbar = figure.colorbar(image, ax=axes.tolist(), shrink=0.95)
        colorbar.set_label(metric_label)

    figure.suptitle(f"Synthetic Experiment 1: {metric_label} Heatmaps", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return [pdf_path, png_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_LABELS.keys()),
        default="rotation_recovery_error_mean",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="synthetic_exp1_rotation_recovery_heatmaps")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = load_results(args.input)
    output_paths = plot_heatmaps(
        payload=payload,
        metric=args.metric,
        output_dir=args.output_dir,
        stem=args.output_stem,
        dpi=args.dpi,
    )

    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
