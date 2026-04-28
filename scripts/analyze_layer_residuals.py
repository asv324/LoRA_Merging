"""Experiment 13: per-layer GPA residual and B-norm spread analysis.

This post-hoc analysis joins two quantities for every LoRA module:

1. GPA final alignment residual from the restored-head Experiment 6 GPA run.
2. Raw per-module ``B``-factor norm spread across source adapters.

It reports Spearman correlations overall and stratified by module family, then
renders a layer-depth x module-family residual heatmap.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

from scripts.merge_task_arithmetic import TASKS, load_adapter_bundle, to_jsonable, validate_compatible_adapters

LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--best-hparams-path", default=PROJECT_ROOT / "results" / "best_hparams.json")
    parser.add_argument(
        "--gpa-result-path",
        default=None,
        help=(
            "Path to the GPA sweep JSON whose module_diagnostics should be used. "
            "Defaults to methods.gpa_baseline.source_path resolved against best_hparams.source_root."
        ),
    )
    parser.add_argument("--results-root", default=PROJECT_ROOT / "results" / "layer_analysis")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--figure-path",
        default=PROJECT_ROOT / "dissertation" / "chapters" / "figures" / "layer_residual_heatmap.pdf",
    )
    parser.add_argument("--skip-plot", action="store_true")
    return parser


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_gpa_result_path(best_hparams_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    payload = load_json(best_hparams_path)
    source_root = payload.get("source_root")
    methods = payload.get("methods", {})
    if not isinstance(source_root, str) or not isinstance(methods, dict):
        raise ValueError(f"Could not resolve source_root/methods from {best_hparams_path}")

    gpa_baseline = methods.get("gpa_baseline")
    if not isinstance(gpa_baseline, dict) or not isinstance(gpa_baseline.get("source_path"), str):
        raise ValueError(f"Could not resolve methods.gpa_baseline.source_path from {best_hparams_path}")

    source_root_path = Path(source_root)
    if not source_root_path.is_absolute():
        source_root_path = PROJECT_ROOT / source_root_path
    return source_root_path / str(gpa_baseline["source_path"])


def parse_layer_index(module_name: str) -> int:
    match = LAYER_PATTERN.search(module_name)
    if match is None:
        raise ValueError(f"Could not parse layer index from module name: {module_name}")
    return int(match.group(1))


def parse_module_family(module_name: str) -> str:
    if ".self_attn." in module_name:
        return "attention"
    if ".mlp." in module_name:
        return "mlp"
    return "other"


def parse_module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def rank_values(values: Sequence[float]) -> List[float]:
    """Return average ranks for Spearman correlation, handling ties."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    cursor = 0
    while cursor < len(indexed):
        next_cursor = cursor + 1
        while next_cursor < len(indexed) and indexed[next_cursor][1] == indexed[cursor][1]:
            next_cursor += 1
        # Ranks are 1-indexed; ties get the average rank.
        average_rank = (cursor + 1 + next_cursor) / 2.0
        for original_index, _ in indexed[cursor:next_cursor]:
            ranks[original_index] = average_rank
        cursor = next_cursor
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_var * y_var)
    if denominator == 0.0:
        return None
    return numerator / denominator


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, object]:
    paired = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if not math.isnan(float(x)) and not math.isnan(float(y)) and math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(paired) < 2:
        return {"rho": None, "n": len(paired)}
    x_values, y_values = zip(*paired)
    rho = pearson(rank_values(x_values), rank_values(y_values))
    return {"rho": rho, "n": len(paired)}


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def load_gpa_diagnostics(gpa_result_path: Path) -> Dict[str, Dict[str, object]]:
    payload = load_json(gpa_result_path)
    diagnostics = payload.get("gpa", {}).get("module_diagnostics", {})  # type: ignore[union-attr]
    if not isinstance(diagnostics, list):
        raise ValueError(f"No gpa.module_diagnostics list found in {gpa_result_path}")
    output = {}
    for entry in diagnostics:
        if not isinstance(entry, dict) or "module_name" not in entry:
            continue
        output[str(entry["module_name"])] = entry
    return output


def b_norm_spread_for_module(adapter_bundles: Sequence[Dict[str, object]], module_name: str) -> Dict[str, object]:
    b_key = f"{module_name}.lora_B.weight"
    norms = []
    for bundle in adapter_bundles:
        tensor = bundle["state_dict"][b_key]
        norms.append(float(tensor.float().norm(p="fro").item()))
    min_norm = min(norms)
    max_norm = max(norms)
    spread = None if min_norm <= 0.0 else max_norm / min_norm
    return {
        "b_frobenius_norms": norms,
        "b_norm_min": min_norm,
        "b_norm_max": max_norm,
        "b_norm_spread": spread,
    }


def build_heatmap_matrix(rows: Sequence[Dict[str, object]]) -> Tuple[List[int], List[str], List[List[float | None]]]:
    layer_indices = sorted({int(row["layer_index"]) for row in rows})
    families = ["attention", "mlp"]
    matrix: List[List[float | None]] = []
    for layer_index in layer_indices:
        matrix_row: List[float | None] = []
        for family in families:
            values = [
                float(row["normalised_alignment_residual"])
                for row in rows
                if int(row["layer_index"]) == layer_index and row["module_family"] == family
            ]
            matrix_row.append(mean(values))
        matrix.append(matrix_row)
    return layer_indices, families, matrix


def grouped_correlations(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {}

    def correlation_for(filter_family: str | None = None) -> Dict[str, object]:
        selected = [
            row
            for row in rows
            if row["b_norm_spread"] is not None
            and (filter_family is None or row["module_family"] == filter_family)
        ]
        return spearman(
            [float(row["normalised_alignment_residual"]) for row in selected],
            [float(row["b_norm_spread"]) for row in selected],
        )

    output["overall"] = correlation_for()
    for family in ("attention", "mlp"):
        output[family] = correlation_for(family)
    return output


def analyse_layers(
    *,
    adapters_dir: Path,
    tasks: Sequence[str],
    gpa_result_path: Path,
) -> Dict[str, object]:
    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    modules = validate_compatible_adapters(adapter_bundles)
    diagnostics_by_module = load_gpa_diagnostics(gpa_result_path)

    rows = []
    missing_diagnostics = []
    for module_name in modules:
        diagnostics = diagnostics_by_module.get(module_name)
        if diagnostics is None:
            missing_diagnostics.append(module_name)
            continue
        norm_payload = b_norm_spread_for_module(adapter_bundles, module_name)
        residual = float(diagnostics["final_alignment_residual"])
        normalised_residual = residual / len(tasks)
        rows.append(
            {
                "module_name": module_name,
                "layer_index": parse_layer_index(module_name),
                "module_family": parse_module_family(module_name),
                "module_type": parse_module_type(module_name),
                "final_alignment_residual": residual,
                "normalised_alignment_residual": normalised_residual,
                "gpa_iterations": diagnostics.get("iterations"),
                "gpa_optimisation_residual": diagnostics.get("optimisation_residual"),
                **norm_payload,
            }
        )

    layer_indices, families, heatmap_matrix = build_heatmap_matrix(rows)
    return {
        "step": "step_4_4_experiment_13_layer_residual_norm_correlation",
        "tasks": list(tasks),
        "adapters_dir": str(adapters_dir),
        "gpa_result_path": str(gpa_result_path),
        "module_count": len(rows),
        "missing_diagnostics": missing_diagnostics,
        "correlations": grouped_correlations(rows),
        "heatmap": {
            "value": "mean_normalised_alignment_residual",
            "layer_indices": layer_indices,
            "families": families,
            "matrix": heatmap_matrix,
        },
        "per_module": sorted(rows, key=lambda row: (int(row["layer_index"]), str(row["module_family"]), str(row["module_type"]))),
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
            "figure.facecolor": "#FAFAFA",
            "savefig.facecolor": "#FAFAFA",
        }
    )


def plot_heatmap(summary: Dict[str, object], figure_path: Path) -> List[Path]:
    if plt is None or np is None:
        raise SystemExit("matplotlib and numpy are required for plotting. Re-run with --skip-plot to write JSON only.")

    heatmap = summary["heatmap"]
    matrix = np.array(heatmap["matrix"], dtype=float)
    layers = heatmap["layer_indices"]
    families = heatmap["families"]

    set_publication_style()
    fig, ax = plt.subplots(figsize=(4.8, 8.0), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("GPA Alignment Residual by Layer")
    ax.set_xlabel("Module family")
    ax.set_ylabel("Layer depth")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels([str(family).upper() for family in families])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    fig.colorbar(image, ax=ax, shrink=0.85, label="Mean residual / adapter")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = figure_path.with_suffix(".png")
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return [figure_path, png_path]


def main() -> None:
    args = build_arg_parser().parse_args()
    best_hparams_path = Path(args.best_hparams_path)
    gpa_result_path = resolve_gpa_result_path(best_hparams_path, args.gpa_result_path)
    results_root = Path(args.results_root)
    summary_path = Path(args.summary_path) if args.summary_path else results_root / "summary.json"

    summary = analyse_layers(
        adapters_dir=Path(args.adapters_dir),
        tasks=list(args.tasks),
        gpa_result_path=gpa_result_path,
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    print(f"Saved layer analysis summary to {summary_path}")

    correlations = summary["correlations"]
    print("Spearman correlations (residual vs B-norm spread):")
    for key in ("overall", "attention", "mlp"):
        block = correlations[key]
        rho = block["rho"]
        rho_text = "n/a" if rho is None else f"{rho:+.4f}"
        print(f"  {key}: rho={rho_text}, n={block['n']}")

    if not args.skip_plot:
        outputs = plot_heatmap(summary, Path(args.figure_path))
        for path in outputs:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
