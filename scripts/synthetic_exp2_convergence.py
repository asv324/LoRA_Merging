"""Synthetic Experiment 2: GPA convergence curves.

Implements the Track A.3 protocol from the revised implementation plan:
- Base configuration: sigma=0.1, N=10, r=16, d=1536
- Record residual at every GPA iteration
- Report iterations to convergence and wall-clock time
- Run an additional sweep over N in {3, 5, 10, 20} and r in {8, 16, 32}
- Save both numeric results and dissertation-ready plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - depends on local environment
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa import gpa_align
from scripts.dissertation_plot_style import (
    COLORS,
    PALETTE,
    add_panel_label,
    apply_style,
    clean_axes,
    figure_size,
    save_figure,
)


def parse_int_list(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def random_orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    gaussian = rng.normal(size=(size, size))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def generate_observed_matrices(
    sigma: float,
    num_adapters: int,
    rank: int,
    dimension: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    shared_matrix = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
    generator_rotations = [random_orthogonal(rng, rank) for _ in range(num_adapters)]
    return [
        rotation @ shared_matrix + sigma * rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
        for rotation in generator_rotations
    ]


def is_monotone_nonincreasing(values: Sequence[float], atol: float = 1e-10) -> bool:
    return all(current <= previous + atol for previous, current in zip(values, values[1:]))


def run_convergence_trial(
    sigma: float,
    num_adapters: int,
    rank: int,
    dimension: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    init: str,
) -> Dict[str, object]:
    observed_matrices = generate_observed_matrices(
        sigma=sigma,
        num_adapters=num_adapters,
        rank=rank,
        dimension=dimension,
        rng=rng,
    )

    started_at = time.perf_counter()
    _, _, residuals = gpa_align(
        observed_matrices,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )
    wall_clock_seconds = time.perf_counter() - started_at

    return {
        "iterations": len(residuals),
        "wall_clock_seconds": float(wall_clock_seconds),
        "residual_trajectory": [float(value) for value in residuals],
        "final_residual": float(residuals[-1]),
        "monotone_nonincreasing": is_monotone_nonincreasing(residuals),
    }


def aggregate_trials(trials: Sequence[Dict[str, object]]) -> Dict[str, object]:
    iteration_values = np.asarray([trial["iterations"] for trial in trials], dtype=float)
    wall_clock_values = np.asarray([trial["wall_clock_seconds"] for trial in trials], dtype=float)
    final_residual_values = np.asarray([trial["final_residual"] for trial in trials], dtype=float)

    max_len = max(len(trial["residual_trajectory"]) for trial in trials)
    residual_matrix = np.full((len(trials), max_len), np.nan, dtype=float)
    for index, trial in enumerate(trials):
        residuals = np.asarray(trial["residual_trajectory"], dtype=float)
        residual_matrix[index, : len(residuals)] = residuals

    return {
        "num_trials": len(trials),
        "iterations_mean": float(iteration_values.mean()),
        "iterations_std": float(iteration_values.std()),
        "iterations_min": float(iteration_values.min()),
        "iterations_max": float(iteration_values.max()),
        "wall_clock_seconds_mean": float(wall_clock_values.mean()),
        "wall_clock_seconds_std": float(wall_clock_values.std()),
        "final_residual_mean": float(final_residual_values.mean()),
        "final_residual_std": float(final_residual_values.std()),
        "all_monotone_nonincreasing": all(bool(trial["monotone_nonincreasing"]) for trial in trials),
        "mean_residual_trajectory": np.nanmean(residual_matrix, axis=0).tolist(),
        "std_residual_trajectory": np.nanstd(residual_matrix, axis=0).tolist(),
    }


def iter_sweep_configs(
    num_adapters_values: Iterable[int],
    ranks: Iterable[int],
) -> Iterable[Tuple[int, int]]:
    for num_adapters in num_adapters_values:
        for rank in ranks:
            yield num_adapters, rank


def plot_convergence_figure(
    baseline_summary: Dict[str, object],
    sweep_summaries: Sequence[Dict[str, object]],
    output_dir: Path,
    dpi: int,
    output_stem: str = "synthetic_exp2_convergence",
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate convergence figures")

    apply_style()
    figure, (ax_base, ax_sweep) = plt.subplots(
        1,
        2,
        figsize=figure_size(180, 82),
        constrained_layout=True,
    )

    baseline_residuals = np.asarray(baseline_summary["mean_residual_trajectory"], dtype=float)
    baseline_iterations = np.arange(1, len(baseline_residuals) + 1)
    ax_base.plot(
        baseline_iterations,
        baseline_residuals,
        marker="o",
        color=COLORS["teal"],
        linewidth=2.0,
    )
    ax_base.set_yscale("log")
    ax_base.set_xlabel("GPA iteration")
    ax_base.set_ylabel("Residual sum of squares")
    ax_base.set_title("Baseline: sigma = 0.1, N = 10, r = 16")
    ax_base.set_xticks(baseline_iterations)
    clean_axes(ax_base, grid_axis="both")
    add_panel_label(ax_base, "a")

    adapter_values = sorted({int(summary["num_adapters"]) for summary in sweep_summaries})
    rank_values = sorted({int(summary["rank"]) for summary in sweep_summaries})
    adapter_colors = {
        adapter_count: PALETTE[index % len(PALETTE)]
        for index, adapter_count in enumerate(adapter_values)
    }
    rank_styles = {
        rank: style
        for rank, style in zip(rank_values, [("o", "-"), ("s", "--"), ("^", ":")])
    }

    for summary in sweep_summaries:
        residuals = np.asarray(summary["mean_residual_trajectory"], dtype=float)
        iterations = np.arange(1, len(residuals) + 1)
        label = f"N={summary['num_adapters']}, r={summary['rank']}"
        marker, linestyle = rank_styles[int(summary["rank"])]
        ax_sweep.plot(
            iterations,
            residuals,
            marker=marker,
            linewidth=1.5,
            linestyle=linestyle,
            color=adapter_colors[int(summary["num_adapters"])],
            label=label,
        )

    ax_sweep.set_yscale("log")
    ax_sweep.set_xlabel("GPA iteration")
    ax_sweep.set_ylabel("Residual sum of squares")
    ax_sweep.set_title("Sweep over adapter count and rank")
    ax_sweep.set_xticks(baseline_iterations)
    clean_axes(ax_sweep, grid_axis="both")
    add_panel_label(ax_sweep, "b")
    ax_sweep.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)

    return save_figure(figure, output_dir, output_stem, dpi=dpi)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--baseline-num-adapters", type=int, default=10)
    parser.add_argument("--baseline-rank", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--sweep-num-adapters", type=parse_int_list, default=parse_int_list("3,5,10,20"))
    parser.add_argument("--sweep-ranks", type=parse_int_list, default=parse_int_list("8,16,32"))
    parser.add_argument("--sweep-trials", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp2_convergence.json",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    master_rng = np.random.default_rng(args.seed)

    baseline_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
    baseline_trial = run_convergence_trial(
        sigma=args.sigma,
        num_adapters=args.baseline_num_adapters,
        rank=args.baseline_rank,
        dimension=args.dimension,
        rng=np.random.default_rng(baseline_seed),
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
    )

    sweep_summaries: List[Dict[str, object]] = []
    for num_adapters, rank in iter_sweep_configs(args.sweep_num_adapters, args.sweep_ranks):
        print(f"Running sweep config sigma={args.sigma}, N={num_adapters}, r={rank}")
        trials = []
        for trial_index in range(args.sweep_trials):
            trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            trials.append(
                run_convergence_trial(
                    sigma=args.sigma,
                    num_adapters=num_adapters,
                    rank=rank,
                    dimension=args.dimension,
                    rng=np.random.default_rng(trial_seed),
                    max_iter=args.max_iter,
                    tol=args.tol,
                    init=args.init,
                )
            )

        sweep_summaries.append(
            {
                "sigma": args.sigma,
                "num_adapters": num_adapters,
                "rank": rank,
                "dimension": args.dimension,
                "tol": args.tol,
                **aggregate_trials(trials),
            }
        )

    baseline_summary = {
        "sigma": args.sigma,
        "num_adapters": args.baseline_num_adapters,
        "rank": args.baseline_rank,
        "dimension": args.dimension,
        "tol": args.tol,
        "mean_residual_trajectory": baseline_trial["residual_trajectory"],
        "iterations_mean": float(baseline_trial["iterations"]),
        "wall_clock_seconds_mean": float(baseline_trial["wall_clock_seconds"]),
        "all_monotone_nonincreasing": bool(baseline_trial["monotone_nonincreasing"]),
        "final_residual_mean": float(baseline_trial["final_residual"]),
    }

    fast_convergence_holds = all(summary["iterations_max"] <= 10 for summary in sweep_summaries)
    payload = {
        "experiment": "synthetic_exp2_convergence",
        "description": "Convergence curves and runtime summaries for GPA on synthetic LoRA-like matrices.",
        "parameters": {
            "sigma": args.sigma,
            "baseline_num_adapters": args.baseline_num_adapters,
            "baseline_rank": args.baseline_rank,
            "dimension": args.dimension,
            "sweep_num_adapters": args.sweep_num_adapters,
            "sweep_ranks": args.sweep_ranks,
            "sweep_trials": args.sweep_trials,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "init": args.init,
            "seed": args.seed,
        },
        "baseline_trial": {
            "seed": baseline_seed,
            **baseline_trial,
        },
        "baseline_summary": baseline_summary,
        "sweep_summaries": sweep_summaries,
        "fast_convergence_check": {
            "description": "All sweep configurations should converge in <= 10 iterations.",
            "passed": fast_convergence_holds,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")
    print(f"Fast convergence check passed: {fast_convergence_holds}")

    if not args.skip_plot:
        output_paths = plot_convergence_figure(
            baseline_summary=baseline_summary,
            sweep_summaries=sweep_summaries,
            output_dir=args.figure_dir,
            dpi=args.dpi,
        )
        for path in output_paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
