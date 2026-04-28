"""Experiment 10: CKA before and after GPA alignment.

Computes pairwise linear CKA between LoRA ``A`` factors for the real GLUE
adapters, first in their raw factor bases and then after applying the same GPA
alignment routine used by ``merge_gpa_ties.py``.

The default output paths follow the dissertation layout used by the current
repo:

- ``results/cka/summary.json``
- ``dissertation/chapters/figures/cka_before_after.pdf``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover - plotting is optional via --skip-plot
    plt = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa_align_adapters import align_module_factors
from scripts.merge_task_arithmetic import TASKS, load_adapter_bundle, to_jsonable, validate_compatible_adapters


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--results-root", default=PROJECT_ROOT / "results" / "cka")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--figure-path",
        default=PROJECT_ROOT / "dissertation" / "chapters" / "figures" / "cka_before_after.pdf",
    )
    parser.add_argument("--max-iter", "--max_iter", dest="max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument(
        "--normalise-a-factors",
        "--normalise_a_factors",
        dest="normalise_a_factors",
        action="store_true",
        help="Use directional GPA, matching the enhanced variants rather than baseline GPA+TIES.",
    )
    parser.add_argument(
        "--max-modules",
        "--max_modules",
        dest="max_modules",
        type=int,
        default=None,
        help="Optional debug limit on the number of LoRA modules to process.",
    )
    parser.add_argument("--skip-plot", action="store_true", help="Only write JSON; do not render the heatmap figure.")
    return parser


def center_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Treat rank rows as observations and center each feature column."""
    matrix = matrix.detach().cpu().float()
    return matrix - matrix.mean(dim=0, keepdim=True)


def linear_cka(matrix_x: torch.Tensor, matrix_y: torch.Tensor) -> float | None:
    """Linear CKA over LoRA rank components.

    ``A`` has shape ``rank x hidden_dim``. For this experiment, each rank
    component is treated as an observation and hidden dimensions as features, so
    GPA row rotations can change the measured similarity.
    """
    x_centered = center_rows(matrix_x)
    y_centered = center_rows(matrix_y)

    xy_norm_sq = torch.linalg.matrix_norm(x_centered @ y_centered.T, ord="fro").pow(2)
    xx_norm = torch.linalg.matrix_norm(x_centered @ x_centered.T, ord="fro")
    yy_norm = torch.linalg.matrix_norm(y_centered @ y_centered.T, ord="fro")
    denominator = xx_norm * yy_norm
    if float(denominator.item()) <= 0.0:
        return None
    value = float((xy_norm_sq / denominator).item())
    # Tiny numerical excursions can occur in float32.
    return max(0.0, min(1.0, value))


def pairwise_cka_matrix(matrices: Sequence[torch.Tensor]) -> List[List[float | None]]:
    size = len(matrices)
    output: List[List[float | None]] = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size):
        output[i][i] = 1.0
        for j in range(i + 1, size):
            value = linear_cka(matrices[i], matrices[j])
            output[i][j] = value
            output[j][i] = value
    return output


def iter_off_diagonal_values(matrix: Sequence[Sequence[float | None]]) -> Iterable[float]:
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i == j or value is None:
                continue
            if math.isnan(value):
                continue
            yield float(value)


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def mean_matrix(matrices: Sequence[Sequence[Sequence[float | None]]]) -> List[List[float | None]]:
    if not matrices:
        return []
    size = len(matrices[0])
    output: List[List[float | None]] = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            values = [
                float(matrix[i][j])
                for matrix in matrices
                if matrix[i][j] is not None and not math.isnan(float(matrix[i][j]))
            ]
            output[i][j] = mean(values)
    return output


def matrix_delta(after: Sequence[Sequence[float | None]], before: Sequence[Sequence[float | None]]) -> List[List[float | None]]:
    output: List[List[float | None]] = []
    for after_row, before_row in zip(after, before):
        output_row: List[float | None] = []
        for after_value, before_value in zip(after_row, before_row):
            if after_value is None or before_value is None:
                output_row.append(None)
            else:
                output_row.append(float(after_value) - float(before_value))
        output.append(output_row)
    return output


def summarise_pairwise(tasks: Sequence[str], before: Sequence[Sequence[float | None]], after: Sequence[Sequence[float | None]]) -> List[Dict[str, object]]:
    rows = []
    for i, task_i in enumerate(tasks):
        for j in range(i + 1, len(tasks)):
            before_value = before[i][j]
            after_value = after[i][j]
            rows.append(
                {
                    "task_i": task_i,
                    "task_j": tasks[j],
                    "cka_before": before_value,
                    "cka_after": after_value,
                    "cka_delta": None
                    if before_value is None or after_value is None
                    else float(after_value) - float(before_value),
                }
            )
    return rows


def analyse_cka(
    *,
    adapters_dir: Path,
    tasks: Sequence[str],
    max_iter: int,
    tol: float,
    init: str,
    normalise_a_factors: bool,
    max_modules: int | None,
) -> Dict[str, object]:
    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    modules = validate_compatible_adapters(adapter_bundles)
    if max_modules is not None:
        modules = modules[:max_modules]

    module_rows = []
    before_matrices = []
    after_matrices = []
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"
        a_matrices = [bundle["state_dict"][a_key] for bundle in adapter_bundles]
        b_matrices = [bundle["state_dict"][b_key] for bundle in adapter_bundles]

        before_matrix = pairwise_cka_matrix(a_matrices)
        aligned_a_matrices, _, diagnostics = align_module_factors(
            a_matrices,
            b_matrices,
            max_iter=max_iter,
            tol=tol,
            init=init,
            normalise=normalise_a_factors,
        )
        after_matrix = pairwise_cka_matrix(aligned_a_matrices)

        average_before = mean(iter_off_diagonal_values(before_matrix))
        average_after = mean(iter_off_diagonal_values(after_matrix))
        module_rows.append(
            {
                "module_name": module_name,
                "average_pairwise_cka_before": average_before,
                "average_pairwise_cka_after": average_after,
                "average_pairwise_cka_delta": None
                if average_before is None or average_after is None
                else average_after - average_before,
                "cka_before": before_matrix,
                "cka_after": after_matrix,
                "gpa_iterations": diagnostics["iterations"],
                "gpa_final_alignment_residual": diagnostics["final_alignment_residual"],
                "gpa_optimisation_residual": diagnostics["optimisation_residual"],
            }
        )
        before_matrices.append(before_matrix)
        after_matrices.append(after_matrix)

    average_before_matrix = mean_matrix(before_matrices)
    average_after_matrix = mean_matrix(after_matrices)
    delta_matrix = matrix_delta(average_after_matrix, average_before_matrix)
    overall_before = mean(iter_off_diagonal_values(average_before_matrix))
    overall_after = mean(iter_off_diagonal_values(average_after_matrix))

    return {
        "step": "step_4_3_experiment_10_cka_before_after_alignment",
        "tasks": list(tasks),
        "adapters_dir": str(adapters_dir),
        "alignment": {
            "method": "dGPA" if normalise_a_factors else "GPA",
            "max_iter": max_iter,
            "tol": tol,
            "init": init,
            "normalise_a_factors": normalise_a_factors,
        },
        "module_count": len(modules),
        "average_pairwise_cka_before": overall_before,
        "average_pairwise_cka_after": overall_after,
        "average_pairwise_cka_delta": None
        if overall_before is None or overall_after is None
        else overall_after - overall_before,
        "average_cka_before_matrix": average_before_matrix,
        "average_cka_after_matrix": average_after_matrix,
        "average_cka_delta_matrix": delta_matrix,
        "pairwise_summary": summarise_pairwise(tasks, average_before_matrix, average_after_matrix),
        "per_module": module_rows,
    }


def set_publication_style() -> None:
    assert plt is not None
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#FAFAFA",
            "savefig.facecolor": "#FAFAFA",
        }
    )


def plot_heatmaps(summary: Dict[str, object], figure_path: Path) -> List[Path]:
    if plt is None or np is None:
        raise SystemExit("matplotlib and numpy are required for plotting. Re-run with --skip-plot to write JSON only.")

    tasks = summary["tasks"]
    before = np.array(summary["average_cka_before_matrix"], dtype=float)
    after = np.array(summary["average_cka_after_matrix"], dtype=float)

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    titles = [
        f"Before GPA\nmean={summary['average_pairwise_cka_before']:.3f}",
        f"After GPA\nmean={summary['average_pairwise_cka_after']:.3f}",
    ]
    for ax, matrix, title in zip(axes, (before, after), titles):
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_yticklabels(tasks)
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="Linear CKA")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = figure_path.with_suffix(".png")
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [figure_path, png_path]


def main() -> None:
    args = build_arg_parser().parse_args()
    results_root = Path(args.results_root)
    summary_path = Path(args.summary_path) if args.summary_path else results_root / "summary.json"

    summary = analyse_cka(
        adapters_dir=Path(args.adapters_dir),
        tasks=list(args.tasks),
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        normalise_a_factors=args.normalise_a_factors,
        max_modules=args.max_modules,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved CKA summary to {summary_path}")
    print(
        "Average pairwise CKA: "
        f"{summary['average_pairwise_cka_before']:.4f} -> {summary['average_pairwise_cka_after']:.4f} "
        f"(delta {summary['average_pairwise_cka_delta']:+.4f})"
    )

    if not args.skip_plot:
        outputs = plot_heatmaps(summary, Path(args.figure_path))
        for path in outputs:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
