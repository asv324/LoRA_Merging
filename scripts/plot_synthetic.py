"""Generate all dissertation-ready synthetic figures.

This script consolidates the outputs from Synthetic Experiments 1-4 into the
four figure types called out in Track A.6 of the revised implementation plan.
It reads the saved JSON artifacts, regenerates the figures, and writes them to
`results/figures/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "matplotlib is required to generate synthetic figures. Install it with `pip install matplotlib`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_synthetic_exp1_heatmaps import annotate_cells, build_heatmap_matrix, extract_axes, load_results
from scripts.synthetic_exp2_convergence import plot_convergence_figure
from scripts.synthetic_exp3_nonorthogonal import plot_robustness_curves
from scripts.synthetic_exp4_structured import plot_structured_overlap


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def plot_rotation_recovery_r16_heatmap(
    payload: Dict[str, object],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    sigmas, num_adapters_values, _ = extract_axes(payload)
    matrix = build_heatmap_matrix(
        summaries=payload["config_summaries"],
        rank=16,
        sigmas=sigmas,
        num_adapters_values=num_adapters_values,
        metric="rotation_recovery_error_mean",
    )

    figure, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    annotate_cells(ax, matrix)
    ax.set_title("Rotation Recovery Heatmap (r = 16)")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Number of adapters N")
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f"{sigma:g}" for sigma in sigmas])
    ax.set_yticks(range(len(num_adapters_values)))
    ax.set_yticklabels([str(value) for value in num_adapters_values])

    colorbar = figure.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("Mean rotation recovery error")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "synthetic_figure1_rotation_recovery_r16.pdf"
    png_path = output_dir / "synthetic_figure1_rotation_recovery_r16.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return [pdf_path, png_path]


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(output_dir: Path, generated_paths: List[Path]) -> Path:
    manifest_path = output_dir / "synthetic_figures_manifest.json"
    manifest_payload = {
        "generated_files": [str(path) for path in generated_paths],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp1-input",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp1.json",
    )
    parser.add_argument(
        "--exp2-input",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp2_convergence.json",
    )
    parser.add_argument(
        "--exp3-input",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp3_nonorthogonal.json",
    )
    parser.add_argument(
        "--exp4-input",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp4_structured.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: List[Path] = []

    exp1_payload = load_results(args.exp1_input)
    generated_paths.extend(plot_rotation_recovery_r16_heatmap(exp1_payload, output_dir=output_dir, dpi=args.dpi))

    exp2_payload = load_json(args.exp2_input)
    generated_paths.extend(
        plot_convergence_figure(
            baseline_summary=exp2_payload["baseline_summary"],
            sweep_summaries=exp2_payload["sweep_summaries"],
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    exp3_payload = load_json(args.exp3_input)
    generated_paths.extend(
        plot_robustness_curves(
            summaries=exp3_payload["config_summaries"],
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    exp4_payload = load_json(args.exp4_input)
    generated_paths.extend(
        plot_structured_overlap(
            summaries=exp4_payload["config_summaries"],
            output_dir=output_dir,
            dpi=args.dpi,
        )
    )

    manifest_path = write_manifest(output_dir=output_dir, generated_paths=generated_paths)
    generated_paths.append(manifest_path)

    for path in generated_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
