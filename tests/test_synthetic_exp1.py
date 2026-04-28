import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_exp1_rotation_recovery import run_single_trial


class SyntheticExperiment1Tests(unittest.TestCase):
    def test_noiseless_trial_recovers_rotations_and_consensus(self) -> None:
        rng = np.random.default_rng(0)
        metrics = run_single_trial(
            sigma=0.0,
            num_adapters=3,
            rank=4,
            dimension=8,
            rng=rng,
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        self.assertLess(metrics["rotation_recovery_error"], 1e-8)
        self.assertLess(metrics["consensus_relative_error"], 1e-8)
        self.assertLess(metrics["alignment_residual"], 1e-8)

    def test_sigma_point_one_remains_small_relative_perturbation(self) -> None:
        rng = np.random.default_rng(1)
        dimension = 1536
        rank = 16

        shared_matrix = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))
        noise_matrix = rng.normal(scale=1.0 / np.sqrt(dimension), size=(rank, dimension))

        relative_noise = 0.1 * np.linalg.norm(noise_matrix, ord="fro") / np.linalg.norm(shared_matrix, ord="fro")
        self.assertLess(relative_noise, 0.2)


if __name__ == "__main__":
    unittest.main()
