"""Synthetic Experiment 4: Structured LoRA-like ground truth.

Implements the Track A.5 protocol from the revised implementation plan:
- Generate A* with geometrically decaying singular values
- Add task-specific rank-1 perturbations P_i
- Apply random rotations Q_i and Gaussian noise
- Measure subspace overlap between span(C) and span(A*)
- Save numeric results and a dissertation-ready overlap figure
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


def sample_unit_vector(rng: np.random.Generator, size: int) -> np.ndarray:
    vector = rng.normal(size=size)
    return vector / (np.linalg.norm(vector) + 1e-12)


def make_structured_shared_matrix(
    rank: int,
    dimension: int,
    decay: float,
    rng: np.random.Generator,
) -> np.ndarray:
    left_basis = random_orthogonal(rng, rank)
    gaussian = rng.normal(size=(dimension, rank))
    right_basis, _ = np.linalg.qr(gaussian)
    singular_values = decay ** np.arange(rank, dtype=float)
    singular_values = singular_values / (np.linalg.norm(singular_values) + 1e-12)
    return left_basis @ np.diag(singular_values) @ right_basis.T


def row_space_basis(matrix: np.ndarray, num_components: int | None = None) -> np.ndarray:
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    effective_rank = int(np.sum(singular_values > 1e-10))
    if num_components is None:
        num_components = effective_rank
    basis_rank = max(1, min(num_components, effective_rank))
    return vh[:basis_rank].T


def principal_angles_degrees(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    singular_values = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return np.degrees(np.arccos(singular_values))


def dominant_subspace_rank(matrix: np.ndarray, energy_threshold: float) -> int:
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    energies = singular_values ** 2
    total_energy = float(np.sum(energies))
    if total_energy <= 0:
        return 1

    cumulative_energy = np.cumsum(energies) / total_energy
    return int(np.searchsorted(cumulative_energy, energy_threshold, side="left") + 1)


def run_single_trial(
    perturbation_strength: float,
    sigma: float,
    num_adapters: int,
    rank: int,
    dimension: int,
    decay: float,
    energy_threshold: float,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    init: str,
) -> Dict[str, float | int | List[float]]:
    shared_matrix = make_structured_shared_matrix(rank=rank, dimension=dimension, decay=decay, rng=rng)
    generator_rotations = [random_orthogonal(rng, rank) for _ in range(num_adapters)]
    observed_matrices = []

    for rotation in generator_rotations:
        left_vector = sample_unit_vector(rng, rank)
        right_vector = sample_unit_vector(rng, dimension)
        task_specific = perturbation_strength * np.outer(left_vector, right_vector)
        noise = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
        observed_matrices.append(rotation @ (shared_matrix + task_specific) + sigma * noise)

    rotations, consensus, residuals = gpa_align(
        observed_matrices,
        max_iter=max_iter,
        tol=tol,
        init=init,
    )

    dominant_rank = dominant_subspace_rank(shared_matrix, energy_threshold=energy_threshold)
    shared_basis = row_space_basis(shared_matrix, num_components=dominant_rank)
    consensus_basis = row_space_basis(consensus, num_components=dominant_rank)
    principal_angles = principal_angles_degrees(consensus_basis, shared_basis)
    cosines = np.cos(np.radians(principal_angles))
    subspace_overlap = float(np.mean(cosines ** 2))
    aligned_matrices = [rotation @ matrix for rotation, matrix in zip(rotations, observed_matrices)]

    return {
        "subspace_overlap": subspace_overlap,
        "dominant_subspace_rank": dominant_rank,
        "mean_principal_angle_deg": float(np.mean(principal_angles)),
        "max_principal_angle_deg": float(np.max(principal_angles)),
        "principal_angles_deg": [float(value) for value in principal_angles],
        "alignment_residual": float(
            np.mean(
                [
                    np.linalg.norm(matrix - consensus, ord="fro") ** 2
                    for matrix in aligned_matrices
                ]
            )
        ),
        "iterations": len(residuals),
        "final_residual": float(residuals[-1]),
    }


def summarize_trials(trials: Sequence[Dict[str, float | int | List[float]]]) -> Dict[str, float | int]:
    metric_names = [
        "subspace_overlap",
        "dominant_subspace_rank",
        "mean_principal_angle_deg",
        "max_principal_angle_deg",
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


def plot_structured_overlap(
    summaries: Sequence[Dict[str, object]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required to generate structured-ground-truth figures")

    strengths = np.asarray([summary["perturbation_strength"] for summary in summaries], dtype=float)
    overlap_means = np.asarray([summary["subspace_overlap_mean"] for summary in summaries], dtype=float)
    overlap_stds = np.asarray([summary["subspace_overlap_std"] for summary in summaries], dtype=float)
    angle_means = np.asarray([summary["mean_principal_angle_deg_mean"] for summary in summaries], dtype=float)
    angle_stds = np.asarray([summary["mean_principal_angle_deg_std"] for summary in summaries], dtype=float)

    figure, (ax_overlap, ax_angle) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    ax_overlap.errorbar(strengths, overlap_means, yerr=overlap_stds, marker="o", linewidth=2, capsize=4)
    ax_overlap.set_xlabel("Task-specific perturbation strength")
    ax_overlap.set_ylabel("Mean dominant-subspace overlap")
    ax_overlap.set_ylim(0.0, 1.02)
    ax_overlap.set_title("Dominant Shared Subspace Overlap")
    ax_overlap.grid(True, linestyle="--", alpha=0.4)

    ax_angle.errorbar(strengths, angle_means, yerr=angle_stds, marker="o", linewidth=2, capsize=4)
    ax_angle.set_xlabel("Task-specific perturbation strength")
    ax_angle.set_ylabel("Mean principal angle (degrees)")
    ax_angle.set_title("Principal Angle vs. Perturbation")
    ax_angle.grid(True, linestyle="--", alpha=0.4)

    figure.suptitle("Synthetic Experiment 4: Structured LoRA-Like Ground Truth", fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "synthetic_exp4_structured.pdf"
    png_path = output_dir / "synthetic_exp4_structured.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return [pdf_path, png_path]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--perturbation-strengths",
        type=parse_float_list,
        default=parse_float_list("0,0.01,0.05,0.1,0.2"),
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--num-adapters", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--energy-threshold", type=float, default=0.9)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_exp4_structured.json",
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

    total_trials = len(args.perturbation_strengths) * args.trials
    completed_trials = 0
    for strength in args.perturbation_strengths:
        print(f"Running perturbation_strength={strength} ({completed_trials}/{total_trials} trials completed)")
        for trial_index in range(args.trials):
            trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            metrics = run_single_trial(
                perturbation_strength=strength,
                sigma=args.sigma,
                num_adapters=args.num_adapters,
                rank=args.rank,
                dimension=args.dimension,
                decay=args.decay,
                energy_threshold=args.energy_threshold,
                rng=np.random.default_rng(trial_seed),
                max_iter=args.max_iter,
                tol=args.tol,
                init=args.init,
            )
            grouped_trials[strength].append(metrics)
            per_trial_results.append(
                {
                    "perturbation_strength": strength,
                    "sigma": args.sigma,
                    "num_adapters": args.num_adapters,
                    "rank": args.rank,
                    "dimension": args.dimension,
                    "decay": args.decay,
                    "energy_threshold": args.energy_threshold,
                    "trial_index": trial_index,
                    "trial_seed": trial_seed,
                    **metrics,
                }
            )
            completed_trials += 1

    summaries = []
    for strength in args.perturbation_strengths:
        summaries.append(
            {
                "perturbation_strength": strength,
                "sigma": args.sigma,
                "num_adapters": args.num_adapters,
                "rank": args.rank,
                "dimension": args.dimension,
                "decay": args.decay,
                "energy_threshold": args.energy_threshold,
                **summarize_trials(grouped_trials[strength]),
            }
        )

    overlap_means = [summary["subspace_overlap_mean"] for summary in summaries]
    shared_subspace_preserved = all(value > 0.9 for value in overlap_means[:4])

    payload = {
        "experiment": "synthetic_exp4_structured",
        "description": "GPA recovery of the shared subspace under structured LoRA-like perturbations.",
        "protocol_note": (
            "A* is built with geometrically decaying singular values and each adapter "
            "receives a task-specific rank-1 perturbation before rotation and noise. "
            "Subspace overlap is measured from the dominant row spaces of the GPA consensus "
            "and the shared ground-truth matrix via principal angles."
        ),
        "parameters": {
            "perturbation_strengths": args.perturbation_strengths,
            "sigma": args.sigma,
            "num_adapters": args.num_adapters,
            "rank": args.rank,
            "dimension": args.dimension,
            "decay": args.decay,
            "energy_threshold": args.energy_threshold,
            "trials_per_config": args.trials,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "init": args.init,
            "seed": args.seed,
        },
        "shared_subspace_check": {
            "description": "Mean dominant-subspace overlap should remain high for low-to-moderate task-specific perturbations.",
            "passed": shared_subspace_preserved,
        },
        "config_summaries": summaries,
        "per_trial_results": per_trial_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")
    print(f"Shared-subspace check passed: {shared_subspace_preserved}")

    if not args.skip_plot:
        output_paths = plot_structured_overlap(summaries=summaries, output_dir=args.figure_dir, dpi=args.dpi)
        for path in output_paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
