"""Plot the Step 4.1 / Experiment 11 N-ablation summary.

Reads the aggregated ``results/ablation_N/summary.json`` produced by
``analyze_ablation_N.py`` and emits the headline figure to
``dissertation/figures/ablation_N.pdf``:

- One line per method, x = N (2, 3, 4, 5), y = mean average primary score
  across the subsets evaluated at that N.
- Error bars: standard deviation across subsets at each N. N = 5 has a
  single subset so no bar is drawn there.
- Style follows ``dissertation_figure_style_guide.md`` (top/right spines
  removed, Inter/Helvetica, teal-led palette, no legend border).

The figure is also saved as PNG alongside the PDF for ad-hoc previewing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "matplotlib is required for plot_ablation_N.py. Install it with `pip install matplotlib`."
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METHOD_ORDER: List[str] = [
    "ties",
    "lr_knots",
    "gpa_baseline",
    "gpa_dgpa_saties_wb_0p5",
]
METHOD_LABELS: Dict[str, str] = {
    "ties": "TIES",
    "lr_knots": "LR-KnOTS+TIES",
    "gpa_baseline": "GPA+TIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA+saTIES+wB(0.5)",
}
# Palette follows the style guide: GPA methods use the signature teal /
# forest green pair; baselines use slate blue / blue-grey.
METHOD_COLORS: Dict[str, str] = {
    "ties": "#5C6BC0",
    "lr_knots": "#78909C",
    "gpa_baseline": "#00BCD4",
    "gpa_dgpa_saties_wb_0p5": "#43A047",
}
METHOD_MARKERS: Dict[str, str] = {
    "ties": "s",
    "lr_knots": "^",
    "gpa_baseline": "o",
    "gpa_dgpa_saties_wb_0p5": "D",
}


def set_publication_style() -> None:
    # Intentionally do not override ``font.family`` / ``font.sans-serif`` here.
    # The remote machine running this plot does not have Inter, Helvetica Neue,
    # or Arial installed, so forcing that stack produced one "findfont: Generic
    # family 'sans-serif' not found" warning per text element drawn before
    # matplotlib fell back to its default ``DejaVu Sans``. Every other figure
    # in this project (``plot_week3_results.py`` etc.) relies on that same
    # default, so omitting the override keeps the N-ablation figure
    # visually consistent with the rest of the dissertation figures and
    # silences the warnings.
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
            "errorbar.capsize": 3,
        }
    )


def load_summary(summary_path: Path) -> Dict[str, object]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def collect_series(
    summary: Dict[str, object],
    method_alias: str,
    n_values: Sequence[int],
) -> Dict[str, List[float]]:
    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []

    cells = summary["cells"][method_alias]  # type: ignore[index]
    for N in n_values:
        cell = cells.get(str(N))
        if cell is None:
            means.append(float("nan"))
            stds.append(0.0)
            counts.append(0)
            continue
        score_block = cell["average_primary_score"]
        means.append(float(score_block["mean"]) if score_block["mean"] is not None else float("nan"))
        std_value = score_block.get("std")
        stds.append(float(std_value) if std_value is not None else 0.0)
        counts.append(int(cell.get("subset_count", 0)))
    return {"means": means, "stds": stds, "counts": counts}


def render_plot(
    summary: Dict[str, object],
    n_values: Sequence[int],
    output_path: Path,
) -> List[Path]:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x_values = np.array(list(n_values), dtype=float)

    for method_alias in METHOD_ORDER:
        if method_alias not in summary["cells"]:  # type: ignore[index]
            continue
        series = collect_series(summary, method_alias, n_values)
        means = np.array(series["means"], dtype=float)
        stds = np.array(series["stds"], dtype=float)

        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            label=METHOD_LABELS[method_alias],
            color=METHOD_COLORS[method_alias],
            marker=METHOD_MARKERS[method_alias],
            markersize=6,
            linewidth=2.0,
            elinewidth=1.0,
        )

    ax.set_xticks(list(n_values))
    ax.set_xlabel("Number of merged adapters (N)")
    ax.set_ylabel("Mean average primary score")
    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.legend(loc="best", handlelength=2.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [output_path, png_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path",
        default=str(PROJECT_ROOT / "results" / "ablation_N" / "summary.json"),
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "dissertation" / "figures" / "ablation_N.pdf"),
    )
    parser.add_argument("--n-values", default="2,3,4,5")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary_path = Path(args.summary_path)
    if not summary_path.exists():
        raise SystemExit(
            f"Summary file not found: {summary_path}. Run scripts/analyze_ablation_N.py first."
        )

    summary = load_summary(summary_path)
    n_values = [int(piece) for piece in args.n_values.split(",") if piece.strip()]
    outputs = render_plot(summary, n_values, Path(args.output_path))
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
