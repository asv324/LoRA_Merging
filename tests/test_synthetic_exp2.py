import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_exp2_convergence import run_convergence_trial


class SyntheticExperiment2Tests(unittest.TestCase):
    def test_residuals_are_monotone_nonincreasing(self) -> None:
        trial = run_convergence_trial(
            sigma=0.1,
            num_adapters=5,
            rank=8,
            dimension=64,
            rng=np.random.default_rng(0),
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        residuals = trial["residual_trajectory"]
        self.assertGreaterEqual(len(residuals), 2)
        self.assertTrue(trial["monotone_nonincreasing"])
        self.assertEqual(trial["iterations"], len(residuals))
        self.assertGreaterEqual(trial["wall_clock_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
