import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_exp4_structured import run_single_trial


class SyntheticExperiment4Tests(unittest.TestCase):
    def test_stronger_task_specific_perturbations_reduce_overlap(self) -> None:
        base_metrics = run_single_trial(
            perturbation_strength=0.0,
            sigma=0.1,
            num_adapters=3,
            rank=4,
            dimension=64,
            decay=0.7,
            energy_threshold=0.9,
            rng=np.random.default_rng(0),
            max_iter=50,
            tol=1e-12,
            init="first",
        )
        perturbed_metrics = run_single_trial(
            perturbation_strength=0.2,
            sigma=0.1,
            num_adapters=3,
            rank=4,
            dimension=64,
            decay=0.7,
            energy_threshold=0.9,
            rng=np.random.default_rng(0),
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        self.assertGreater(
            base_metrics["subspace_overlap"],
            perturbed_metrics["subspace_overlap"],
        )
        self.assertLess(
            base_metrics["mean_principal_angle_deg"],
            perturbed_metrics["mean_principal_angle_deg"],
        )

    def test_no_perturbation_keeps_dominant_overlap_high(self) -> None:
        metrics = run_single_trial(
            perturbation_strength=0.0,
            sigma=0.0,
            num_adapters=3,
            rank=4,
            dimension=64,
            decay=0.7,
            energy_threshold=0.9,
            rng=np.random.default_rng(1),
            max_iter=50,
            tol=1e-12,
            init="first",
        )

        self.assertGreater(metrics["subspace_overlap"], 0.95)
        self.assertLess(metrics["alignment_residual"], 1e-8)


if __name__ == "__main__":
    unittest.main()
