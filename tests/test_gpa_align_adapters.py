import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa_align_adapters import align_module_factors


class GpaAlignAdaptersTests(unittest.TestCase):
    def test_align_module_factors_preserves_effective_delta(self) -> None:
        a_matrices = [
            torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        ]
        b_matrices = [
            torch.tensor([[1.0, 0.0], [0.5, 2.0]], dtype=torch.float32),
            torch.tensor([[0.5, 1.0], [2.0, 1.5]], dtype=torch.float32),
        ]

        aligned_a, aligned_b, diagnostics = align_module_factors(
            a_matrices,
            b_matrices,
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        self.assertGreaterEqual(diagnostics["iterations"], 1)
        for original_a, original_b, new_a, new_b, adapter_diag in zip(
            a_matrices,
            b_matrices,
            aligned_a,
            aligned_b,
            diagnostics["per_adapter"],
        ):
            torch.testing.assert_close(new_b @ new_a, original_b @ original_a, atol=1e-5, rtol=1e-5)
            self.assertLess(adapter_diag["norm_difference"], 1e-5)
            self.assertLess(adapter_diag["functional_invariance_error"], 1e-5)

    def test_align_module_factors_supports_directional_normalisation(self) -> None:
        a_matrices = [
            torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, -6.0], [3.0, 0.0]], dtype=torch.float32),
        ]
        b_matrices = [
            torch.tensor([[1.0, 0.0], [0.5, 2.0]], dtype=torch.float32),
            torch.tensor([[0.5, 1.0], [2.0, 1.5]], dtype=torch.float32),
        ]

        aligned_a, aligned_b, diagnostics = align_module_factors(
            a_matrices,
            b_matrices,
            max_iter=50,
            tol=1e-12,
            init="mean",
            normalise=True,
        )

        self.assertTrue(diagnostics["normalised_alignment"])
        self.assertGreaterEqual(diagnostics["iterations"], 1)
        for original_a, original_b, new_a, new_b, adapter_diag in zip(
            a_matrices,
            b_matrices,
            aligned_a,
            aligned_b,
            diagnostics["per_adapter"],
        ):
            torch.testing.assert_close(new_b @ new_a, original_b @ original_a, atol=1e-5, rtol=1e-5)
            self.assertLess(adapter_diag["norm_difference"], 1e-5)
            self.assertLess(adapter_diag["functional_invariance_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
