import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_cola_prediction_distributions import summarize_method_records


class CollectColaPredictionDistributionsTests(unittest.TestCase):
    def test_summarize_method_records_finds_non_degenerate_threshold(self) -> None:
        records = [
            {
                "source_path": "gpa_ties/baseline/trim_10/lambda_0p05.json",
                "lambda": 0.05,
                "cola_metric": 0.0,
                "prediction_distribution_available": True,
                "degenerate_prediction": True,
            },
            {
                "source_path": "gpa_ties/baseline/trim_10/lambda_0p1.json",
                "lambda": 0.1,
                "cola_metric": 0.2,
                "prediction_distribution_available": True,
                "degenerate_prediction": False,
            },
        ]

        summary = summarize_method_records(records)

        self.assertTrue(summary["prediction_distribution_available"])
        self.assertEqual(summary["non_degenerate_lambda_threshold"], 0.1)
        self.assertFalse(summary["rerun_required"])

    def test_summarize_method_records_marks_missing_distributions(self) -> None:
        records = [
            {
                "source_path": "ties/trim_10/lambda_0p1.json",
                "lambda": 0.1,
                "cola_metric": 0.0,
                "prediction_distribution_available": False,
                "degenerate_prediction": None,
            }
        ]

        summary = summarize_method_records(records)

        self.assertFalse(summary["prediction_distribution_available"])
        self.assertTrue(summary["rerun_required"])


if __name__ == "__main__":
    unittest.main()
