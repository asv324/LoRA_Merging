"""Synthetic Experiment 3: Robustness to non-orthogonal perturbations.

Implements the Track A.4 protocol from the revised implementation plan:
- Generate A_i = (Q_i + delta S_i) A* + sigma E_i
- Sweep delta in {0, 0.01, 0.05, 0.1, 0.2}
- Fix sigma=0.1, N=5, r=16, d=1536
- Measure alignment residual and rotation recovery error
- Save numeric results and the dissertation-ready robustness curve
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def parse_float_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def random_orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    gaussian = rng.normal(size=(size, size))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def sample_unit_frobenius_matrix(
    rng: np.random.Generator,
    shape: Tuple[int, int],
) -> np.ndarray:
    matrix = rng.normal(size=shape)
    return matrix / (np.linalg.norm(matrix, ord="fro") + 1e-12)


def solve_orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cross_covariance = target @ source.T
    u, _, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vt


def resolve_global_rotation(
    estimated_aligners: Sequence[np.ndarray],
    true_aligners: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    estimated_stack = np.concatenate(estimated_aligners, axis=1)
    true_stack = np.concatenate(true_aligners, axis=1)
    global_rotation = solve_orthogonal_procrustes(estimated_stack, true_stack)
    aligned_estimates = [global_rotation @ rotation for rotation in estimated_aligners]
    return global_rotation, aligned_estimates


def run_single_trial(
    delta: float,
    sigma: float,
    num_adapters: int,
    rank: int,
    dimension: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    init: str,
) -> Dict[str, float | int | List[float]]:
    shared_matrix = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
    generator_rotations = [random_orthogonal(rng, rank) for _ in range(num_adapters)]
    perturbations = [sample_unit_frobenius_matrix(rng, (rank, rank)) for _ in range(num_adapters)]
    observed_matrices = []

    for rotation, perturbation in zip(generator_rotations, perturbations):
        transform = rotation + delta * perturbation
        noise = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
        observed_matrices.append(transform @ shared_matrix + sigma * noise)

    estimated_aligners, consensus, residuals = gpa_align(
        observed_matrices,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )

    # We still score against the underlying orthogonal component Q_i^T to
    # quantify how the departure from orthogonality hurts rotation recovery.
    true_aligners = [rotation.T for rotation in generator_rotations]
    global_rotation, aligned_estimates = resolve_global_rotation(estimated_aligners, true_aligners)
    aligned_consensus = global_rotation @ consensus

    rotation_recovery_error = float(
        np.mean(
            [
                np.linalg.norm(estimate - truth, ord="fro") ** 2
                for estimate, truth in zip(aligned_estimates, true_aligners)
            ]
        )
    )
    alignment_residual = float(
        np.mean(
            [
                np.linalg.norm(rotation @ matrix - consensus, ord="fro") ** 2
                for rotation, matrix in zip(estimated_aligners, observed_matrices)
            ]
        )
    )
    consensus_relative_error = float(
        np.linalg.norm(aligned_consensus - shared_matrix, ord="fro")
        / (np.linalg.norm(shared_matrix, ord="fro") + 1e-12)
    )

    return {
        "rotation_recovery_error": rotation_recovery_error,
        "alignment_residual": alignment_residual,
        "consensus_relative_error": consensus_relative_error,
        "iterations": len(residuals),
        "final_residual": float(residuals[-1]),
        "residual_trajectory": [float(value) for value in residuals],
        "global_rotation_det": float(np.linalg.det(global_rotation)),
    }


def summarize_trials(trials: Sequence[Dict[str, float | int | List[float]]]) -> Dict[str, float | int]:
    metric_names = [
        "rotation_recovery_error",
        "alignment_residual",
        "consensus_relative_error",
        "iterations",
        "final_residual",
    ]

    summary: Dict[str, float | int] = {"num_trials": len(trials)}
    for metric_name in metric_names:
        values = np.asarray([trial[metric_name] for trial in trials], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std())
        summary[f"{metric_name}_min"] = float(values.min())
        summary[f"{metric_name}_max"] = float(values.max())

    return summary


def plot_robustness_curves(
    summaries: Sequence[Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate robustness figures")

    deltas = np.asarray([summary["delta"] for summary in summaries], dtype=float)
    residual_means = np.asarray([summary["alignment_residual_mean"] for summary in summaries], dtype=float)
    residual_stds = np.asarray([summary["alignment_residual_std"] for summary in summaries], dtype=float)
    rotation_means = np.asarray([summary["rotation_recovery_error_mean"] for summary in summaries], dtype=float)
    rotation_stds = np.asarray([summary["rotation_recovery_error_std"] for summary in summaries], dtype=float)

    figure, (ax_residual, ax_rotation) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax_residual.errorbar(deltas, residual_means, yerr=residual_stds, marker="o", linewidth=2, capsize=4)
    ax_residual.set_xlabel("Non-orthogonality delta")
    ax_residual.set_ylabel("Mean alignment residual")
    ax_residual.set_title("Alignment Residual vs. Delta")
    ax_residual.grid(True, linestyle="--", alpha=0.4)

    ax_rotation.errorbar(deltas, rotation_means, yerr=rotation_stds, marker="o", linewidth=2, capsize=4)
    ax_rotation.set_xlabel("Non-orthogonality delta")
    ax_rotation.set_ylabel("Mean rotation recovery error")
    ax_rotation.set_title("Rotation Recovery Error vs. Delta")
    ax_rotation.grid(True, linestyle="--", alpha=0.4)

    figure.suptitle("Synthetic Experiment 3: Robustness to Non-Orthogonal Perturbations", fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "synthetic_exp3_nonorthogonal.pdf"
    png_path = output_dir / "synthetic_exp3_nonorthogonal.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return [pdf_path, png_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deltas", type=parse_float_list, default=parse_float_list("0,0.01,0.05,0.1,0.2"))
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--num-adapters", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp3_nonorthogonal.json",
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
    grouped_trials: Dict[float, List[Dict[str, float | int | List[float]]]] = defaultdict(list)
    per_trial_results: List[Dict[str, object]] = []

    total_trials = len(args.deltas) * args.trials
    completed_trials = 0
    for delta in args.deltas:
        print(f"Running delta={delta} ({completed_trials}/{total_trials} trials completed)")
        for trial_index in range(args.trials):
            trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            metrics = run_single_trial(
                delta=delta,
                sigma=args.sigma,
                num_adapters=args.num_adapters,
                rank=args.rank,
                dimension=args.dimension,
                rng=np.random.default_rng(trial_seed),
                max_iter=args.max_iter,
                tol=args.tol,
                init=args.init,
            )
            grouped_trials[delta].append(metrics)
            per_trial_results.append(
                {
                    "delta": delta,
                    "sigma": args.sigma,
                    "num_adapters": args.num_adapters,
                    "rank": args.rank,
                    "dimension": args.dimension,
                    "trial_index": trial_index,
                    "trial_seed": trial_seed,
                    **metrics,
                }
            )
            completed_trials += 1

    summaries = []
    for delta in args.deltas:
        summaries.append(
            {
                "delta": delta,
                "sigma": args.sigma,
                "num_adapters": args.num_adapters,
                "rank": args.rank,
                "dimension": args.dimension,
                **summarize_trials(grouped_trials[delta]),
            }
        )

    residual_means = [summary["alignment_residual_mean"] for summary in summaries]
    graceful_growth = all(
        current <= previous * 2.5 + 1e-12
        for previous, current in zip(residual_means, residual_means[1:])
    )

    payload = {
        "experiment": "synthetic_exp3_nonorthogonal",
        "description": "Robustness of GPA when the shared orthogonality assumption is perturbed.",
        "protocol_note": (
            "Observed matrices are generated as A_i = (Q_i + delta S_i) A* + sigma E_i, "
            "with S_i normalized to unit Frobenius norm. Rotation recovery is scored "
            "against the underlying orthogonal component Q_i^T after resolving one "
            "shared global rotation ambiguity."
        ),
        "parameters": {
            "deltas": args.deltas,
            "sigma": args.sigma,
            "num_adapters": args.num_adapters,
            "rank": args.rank,
            "dimension": args.dimension,
            "trials_per_config": args.trials,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "init": args.init,
            "seed": args.seed,
        },
        "robustness_check": {
            "description": "Alignment residual should grow gracefully rather than exploding as delta increases.",
            "passed": graceful_growth,
        },
        "config_summaries": summaries,
        "per_trial_results": per_trial_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")
    print(f"Graceful-growth check passed: {graceful_growth}")

    if not args.skip_plot:
        output_paths = plot_robustness_curves(summaries=summaries, output_dir=args.figure_dir, dpi=args.dpi)
        for path in output_paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
