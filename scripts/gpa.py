"""Generalized Procrustes Analysis (GPA) for LoRA factor alignment.

Implements Gower-style alternating optimization:
1. Fix consensus C, update rotations Q_i via Procrustes on C @ A_i.T
2. Fix rotations Q_i, update consensus C = mean(Q_i @ A_i)
Repeat until convergence.

All matrices are expected to have shape (r, d).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:
    from scipy.linalg import svd as scipy_svd
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    scipy_svd = None


def _svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use SciPy's SVD when available, otherwise fall back to NumPy."""
    if scipy_svd is not None:
        return scipy_svd(matrix, full_matrices=False)
    return np.linalg.svd(matrix, full_matrices=False)


def _validate_matrices(matrices: Sequence[np.ndarray]) -> Tuple[int, int]:
    if not matrices:
        raise ValueError("matrices must contain at least one array")

    first = np.asarray(matrices[0], dtype=float)
    if first.ndim != 2:
        raise ValueError("each matrix must be 2D")

    shape = first.shape
    for matrix in matrices[1:]:
        array = np.asarray(matrix, dtype=float)
        if array.shape != shape:
            raise ValueError(f"shape mismatch: expected {shape}, got {array.shape}")

    return shape


def _normalise_matrix_frobenius(matrix: np.ndarray) -> np.ndarray:
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    if frobenius_norm <= 0.0:
        raise ValueError("cannot normalise a matrix with zero Frobenius norm")
    return matrix / frobenius_norm


def gpa_align(
    matrices: Sequence[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    init: str = "first",
    verbose: bool = False,
    normalise: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, List[float]]:
    """Align matrices with generalized Procrustes analysis.

    Args:
        matrices: Sequence of arrays, each with shape (r, d).
        max_iter: Maximum number of GPA iterations.
        tol: Relative residual change threshold for convergence.
        init: Consensus initialization strategy, either "first" or "mean".
        verbose: If true, print per-iteration diagnostics.
        normalise: If true, fit rotations on per-matrix unit-Frobenius copies so
            the alignment objective is insensitive to adapter scale.

    Returns:
        rotations: Orthogonal matrices Q_i, each with shape (r, r).
        consensus: Final consensus matrix with shape (r, d).
        residuals: Residual sum-of-squares values for each iteration.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    r, _ = _validate_matrices(matrices)
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    working_arrays = [_normalise_matrix_frobenius(matrix) for matrix in arrays] if normalise else arrays

    if init == "first":
        consensus = working_arrays[0].copy()
    elif init == "mean":
        consensus = np.mean(np.stack(working_arrays, axis=0), axis=0)
    else:
        raise ValueError(f"unknown init strategy: {init}")

    rotations = [np.eye(r, dtype=float) for _ in working_arrays]
    residuals: List[float] = []

    for iteration in range(max_iter):
        for index, matrix in enumerate(working_arrays):
            cross_covariance = consensus @ matrix.T
            u, _, vt = _svd(cross_covariance)
            rotations[index] = u @ vt

        aligned = [rotation @ matrix for rotation, matrix in zip(rotations, working_arrays)]
        consensus = np.mean(np.stack(aligned, axis=0), axis=0)

        residual = float(
            sum(np.linalg.norm(aligned_matrix - consensus, ord="fro") ** 2 for aligned_matrix in aligned)
        )
        residuals.append(residual)

        if verbose:
            print(f"Iter {iteration + 1}: residual={residual:.8e}")

        if iteration > 0:
            previous = residuals[-2]
            rel_change = abs(previous - residuals[-1]) / (previous + 1e-12)
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

    return rotations, consensus, residuals
