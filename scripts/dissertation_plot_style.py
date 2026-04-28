"""Shared plotting and table helpers for dissertation result artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 88
FULL_WIDTH_MM = 180

COLORS = {
    "teal": "#00BCD4",
    "slate": "#5C6BC0",
    "coral": "#EF5350",
    "amber": "#FFA726",
    "green": "#43A047",
    "violet": "#AB47BC",
    "blue_grey": "#78909C",
    "light_grey": "#ECEFF1",
    "grid": "#CFD8DC",
    "charcoal": "#37474F",
    "background": "#FAFAFA",
    "reference": "#9E9E9E",
}

PALETTE = [
    COLORS["teal"],
    COLORS["slate"],
    COLORS["coral"],
    COLORS["amber"],
    COLORS["green"],
    COLORS["violet"],
    COLORS["blue_grey"],
]

TASK_LABELS = {
    "sst2": "SST-2",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "cola": "CoLA",
    "rte": "RTE",
}

TASK_ORDER = ["sst2", "mnli", "qnli", "cola", "rte"]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "deepmind_sequential",
    ["#1A237E", "#0288D1", COLORS["teal"], "#80DEEA", "#FFFFFF"],
)


def mm_to_in(mm: float) -> float:
    """Convert millimetres to inches for matplotlib sizing."""

    return mm / MM_PER_INCH


def figure_size(width_mm: float = FULL_WIDTH_MM, height_mm: float = 95) -> tuple[float, float]:
    """Return a dissertation-sized figure in inches."""

    return mm_to_in(width_mm), mm_to_in(height_mm)


def apply_style() -> None:
    """Apply the dissertation figure style guide to matplotlib."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.edgecolor": COLORS["charcoal"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "errorbar.capsize": 3,
            "legend.frameon": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.5,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    """Apply spine, tick, and grid polish to one axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["charcoal"])
    ax.spines["bottom"].set_color(COLORS["charcoal"])
    ax.tick_params(axis="both", colors=COLORS["charcoal"], length=3)
    ax.xaxis.label.set_color(COLORS["charcoal"])
    ax.yaxis.label.set_color(COLORS["charcoal"])
    ax.title.set_color(COLORS["charcoal"])
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, linestyle=(0, (4, 4)), color=COLORS["grid"], linewidth=0.5, alpha=0.7)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Place a bold lower-case panel label in guide style."""

    ax.text(
        -0.08,
        1.05,
        f"({label})",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color=COLORS["charcoal"],
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, *, dpi: int = 300) -> list[Path]:
    """Save PDF and PNG versions of a figure and close it."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return [pdf_path, png_path]


def metric(value: float | int | None, digits: int = 3) -> str:
    """Format numeric table values consistently."""

    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def latex_escape(value: object) -> str:
    """Escape a small amount of LaTeX special syntax for table cells."""

    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write a CSV table with a stable column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_latex_table(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    *,
    caption: str,
    label: str,
) -> None:
    """Write a compact LaTeX tabular environment for dissertation tables."""

    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = "l" + "r" * (len(fieldnames) - 1)
    header = " & ".join(latex_escape(field) for field in fieldnames)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        rf"{header} \\",
        r"\midrule",
    ]
    for row in rows:
        body = " & ".join(latex_escape(row.get(field, "")) for field in fieldnames)
        lines.append(rf"{body} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def relative_paths(paths: Iterable[Path], root: Path) -> list[str]:
    """Return POSIX-style relative paths for manifests."""

    return [path.resolve().relative_to(root.resolve()).as_posix() for path in paths]
