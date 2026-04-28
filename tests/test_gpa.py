import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa import gpa_align


class GpaAlignTests(unittest.TestCase):
    def test_identity_rotations_return_zero_residual(self) -> None:
        rng = np.random.default_rng(0)
        base_matrix = rng.normal(size=(4, 8))
        matrices = [base_matrix.copy() for _ in range(3)]

        rotations, consensus, residuals = gpa_align(
            matrices,
            max_iter=10,
            tol=1e-12,
            init="first",
        )

        self.assertGreaterEqual(len(residuals), 1)

        for rotation in rotations:
            np.testing.assert_allclose(rotation, np.eye(4), atol=1e-10)

        np.testing.assert_allclose(consensus, base_matrix, atol=1e-10)
        self.assertAlmostEqual(residuals[-1], 0.0, places=10)

    def test_normalised_alignment_is_invariant_to_per_matrix_scaling(self) -> None:
        theta = np.deg2rad(35.0)
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        base_matrix = np.array(
            [
                [1.0, 2.0, -1.0],
                [0.5, -0.25, 3.0],
            ]
        )

        reference_rotations, _, _ = gpa_align(
            [base_matrix, rotation @ base_matrix],
            max_iter=50,
            tol=1e-12,
            init="mean",
            normalise=True,
        )
        scaled_rotations, _, _ = gpa_align(
            [base_matrix, 9.0 * rotation @ base_matrix],
            max_iter=50,
            tol=1e-12,
            init="mean",
            normalise=True,
        )

        for reference, scaled in zip(reference_rotations, scaled_rotations):
            np.testing.assert_allclose(reference, scaled, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
