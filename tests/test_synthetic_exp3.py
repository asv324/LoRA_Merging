import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_exp3_nonorthogonal import run_single_trial


class SyntheticExperiment3Tests(unittest.TestCase):
    def test_nonorthogonal_perturbation_increases_difficulty(self) -> None:
        base_metrics = run_single_trial(
            delta=0.0,
            sigma=0.1,
            num_adapters=3,
            rank=4,
            dimension=64,
            rng=np.random.default_rng(0),
            max_iter=50,
            tol=1e-12,
            init="first",
        )
        perturbed_metrics = run_single_trial(
            delta=0.2,
            sigma=0.1,
            num_adapters=3,
            rank=4,
            dimension=64,
            rng=np.random.default_rng(0),
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        self.assertGreater(
            perturbed_metrics["alignment_residual"],
            base_metrics["alignment_residual"],
        )
        self.assertGreater(
            perturbed_metrics["rotation_recovery_error"],
            base_metrics["rotation_recovery_error"],
        )


if __name__ == "__main__":
    unittest.main()
