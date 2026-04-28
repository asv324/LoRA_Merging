import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_synthetic_exp1_heatmaps import build_heatmap_matrix


class PlotSyntheticTests(unittest.TestCase):
    def test_rotation_recovery_heatmap_matrix_for_rank_16_has_expected_shape(self) -> None:
        payload = {
            "config_summaries": [
                {"sigma": 0.0, "num_adapters": 3, "rank": 16, "rotation_recovery_error_mean": 1.0},
                {"sigma": 0.1, "num_adapters": 3, "rank": 16, "rotation_recovery_error_mean": 2.0},
                {"sigma": 0.0, "num_adapters": 5, "rank": 16, "rotation_recovery_error_mean": 3.0},
                {"sigma": 0.1, "num_adapters": 5, "rank": 16, "rotation_recovery_error_mean": 4.0},
            ]
        }

        matrix = build_heatmap_matrix(
            summaries=payload["config_summaries"],
            rank=16,
            sigmas=[0.0, 0.1],
            num_adapters_values=[3, 5],
            metric="rotation_recovery_error_mean",
        )

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[1, 1], 4.0)


if __name__ == "__main__":
    unittest.main()
