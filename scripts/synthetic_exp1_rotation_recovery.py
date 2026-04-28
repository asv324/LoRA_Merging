"""Synthetic Experiment 1: GPA ground-truth rotation recovery.

This script implements the Track A.2 protocol from the revised implementation
plan. It generates synthetic LoRA-like matrices, runs GPA alignment, resolves
the global rotation ambiguity inherent to GPA, and stores full per-trial
results to JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa import gpa_align


def parse_float_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def random_orthogonal(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample an orthogonal matrix using QR decomposition of a Gaussian draw."""
    gaussian = rng.normal(size=(size, size))
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def solve_orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R minimizing ||R @ source - target||_F."""
    cross_covariance = target @ source.T
    u, _, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vt


def resolve_global_rotation(
    estimated_aligners: Sequence[np.ndarray],
    true_aligners: Sequence[np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Align estimated rotations to ground truth with one shared orthogonal map."""
    estimated_stack = np.concatenate(estimated_aligners, axis=1)
    true_stack = np.concatenate(true_aligners, axis=1)
    global_rotation = solve_orthogonal_procrustes(estimated_stack, true_stack)
    aligned_estimates = [global_rotation @ rotation for rotation in estimated_aligners]
    return global_rotation, aligned_estimates


def run_single_trial(
    sigma: float,
    num_adapters: int,
    rank: int,
    dimension: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    init: str,
) -> Dict[str, float | int | List[float]]:
    """Run one synthetic GPA trial and return metrics plus diagnostics."""
    shared_matrix = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
    generator_rotations = [random_orthogonal(rng, rank) for _ in range(num_adapters)]
    observed_matrices = [
        # Match the entry scale of A* so sigma controls relative noise level.
        rotation @ shared_matrix + sigma * rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
        for rotation in generator_rotations
    ]

    estimated_aligners, consensus, residuals = gpa_align(
        observed_matrices,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )

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
    consensus_relative_error = float(
        np.linalg.norm(aligned_consensus - shared_matrix, ord="fro")
        / (np.linalg.norm(shared_matrix, ord="fro") + 1e-12)
    )
    alignment_residual = float(
        np.mean(
            [
                np.linalg.norm(rotation @ matrix - consensus, ord="fro") ** 2
                for rotation, matrix in zip(estimated_aligners, observed_matrices)
            ]
        )
    )

    return {
        "rotation_recovery_error": rotation_recovery_error,
        "consensus_relative_error": consensus_relative_error,
        "alignment_residual": alignment_residual,
        "iterations": len(residuals),
        "final_residual": float(residuals[-1]),
        "residual_trajectory": [float(value) for value in residuals],
        "global_rotation_det": float(np.linalg.det(global_rotation)),
    }


def summarize_trials(trials: Sequence[Dict[str, float | int | List[float]]]) -> Dict[str, float | int]:
    metric_names = [
        "rotation_recovery_error",
        "consensus_relative_error",
        "alignment_residual",
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


def iter_configs(
    sigmas: Iterable[float],
    num_adapters_values: Iterable[int],
    ranks: Iterable[int],
) -> Iterable[Tuple[float, int, int]]:
    for sigma in sigmas:
        for num_adapters in num_adapters_values:
            for rank in ranks:
                yield sigma, num_adapters, rank


def run_sweep(args: argparse.Namespace) -> Dict[str, object]:
    master_rng = np.random.default_rng(args.seed)
    started_at = time.perf_counter()
    per_trial_results: List[Dict[str, object]] = []
    grouped_trials: Dict[Tuple[float, int, int], List[Dict[str, float | int | List[float]]]] = defaultdict(list)

    configs = list(iter_configs(args.sigmas, args.num_adapters, args.ranks))
    total_trials = len(configs) * args.trials
    completed_trials = 0

    for sigma, num_adapters, rank in configs:
        print(
            f"Running sigma={sigma}, N={num_adapters}, r={rank} "
            f"({completed_trials}/{total_trials} trials completed)"
        )
        for trial_index in range(args.trials):
            trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            trial_rng = np.random.default_rng(trial_seed)
            metrics = run_single_trial(
                sigma=sigma,
                num_adapters=num_adapters,
                rank=rank,
                dimension=args.dimension,
                rng=trial_rng,
                max_iter=args.max_iter,
                tol=args.tol,
                init=args.init,
            )
            grouped_trials[(sigma, num_adapters, rank)].append(metrics)
            per_trial_results.append(
                {
                    "sigma": sigma,
                    "num_adapters": num_adapters,
                    "rank": rank,
                    "dimension": args.dimension,
                    "trial_index": trial_index,
                    "trial_seed": trial_seed,
                    **metrics,
                }
            )
            completed_trials += 1

    summaries = []
    for sigma, num_adapters, rank in configs:
        trials = grouped_trials[(sigma, num_adapters, rank)]
        summaries.append(
            {
                "sigma": sigma,
                "num_adapters": num_adapters,
                "rank": rank,
                "dimension": args.dimension,
                **summarize_trials(trials),
            }
        )

    critical_configs = [
        summary
        for summary in summaries
        if summary["rank"] == 16 and summary["num_adapters"] == 5 and summary["sigma"] <= 0.1
    ]
    critical_success = bool(critical_configs) and all(
        summary["rotation_recovery_error_mean"] < 0.01 for summary in critical_configs
    )

    return {
        "experiment": "synthetic_exp1_rotation_recovery",
        "description": "Ground-truth rotation recovery for GPA on synthetic LoRA-like matrices.",
        "protocol_note": (
            "Observed matrices are generated as A_i = Q_i A* + sigma E_i. "
            "Both A* and E_i use entry scale 1/sqrt(d), so sigma directly controls "
            "the relative perturbation magnitude instead of growing with dimension. "
            "Because GPA estimates aligners that map A_i back to consensus, "
            "metrics compare estimated rotations to Q_i^T after resolving one "
            "shared global orthogonal ambiguity."
        ),
        "parameters": {
            "sigmas": args.sigmas,
            "num_adapters": args.num_adapters,
            "ranks": args.ranks,
            "dimension": args.dimension,
            "trials_per_config": args.trials,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "init": args.init,
            "seed": args.seed,
        },
        "critical_success_criterion": {
            "description": "For sigma <= 0.1, N = 5, r = 16, mean rotation recovery error should be < 0.01.",
            "passed": critical_success,
            "evaluated_configs": critical_configs,
        },
        "wall_clock_seconds": float(time.perf_counter() - started_at),
        "config_summaries": summaries,
        "per_trial_results": per_trial_results,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigmas", type=parse_float_list, default=parse_float_list("0,0.01,0.05,0.1,0.2,0.5"))
    parser.add_argument("--num-adapters", type=parse_int_list, default=parse_int_list("3,5,10"))
    parser.add_argument("--ranks", type=parse_int_list, default=parse_int_list("4,8,16,32"))
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp1.json",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_sweep(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved results to {args.output}")
    print(f"Critical success criterion passed: {payload['critical_success_criterion']['passed']}")


if __name__ == "__main__":
    main()
