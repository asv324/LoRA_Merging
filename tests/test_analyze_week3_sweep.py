import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_week3_sweep import (
    METHOD_LABELS,
    SweepRecord,
    build_best_hparams_payload,
    build_gpa_rerun_decision_payload,
)


def make_record(
    *,
    method_key: str,
    avg: float,
    cola: float,
    source_path: str,
    gpa_fraction: float | None = None,
) -> SweepRecord:
    gpa_module_count = None
    gpa_modules_hitting_max_iter = None
    if gpa_fraction is not None:
        gpa_module_count = 10
        gpa_modules_hitting_max_iter = int(round(gpa_fraction * gpa_module_count))

    return SweepRecord(
        method_key=method_key,
        display_name=METHOD_LABELS[method_key],
        source_path=source_path,
        average_primary_score=avg,
        primary_metrics={
            "sst2": {"metric": "accuracy", "value": avg},
            "mnli": {"metric": "accuracy", "value": avg},
            "qnli": {"metric": "accuracy", "value": avg},
            "cola": {"metric": "matthews_correlation", "value": cola},
            "rte": {"metric": "accuracy", "value": avg},
        },
        evaluation={},
        lambda_value=0.2,
        trim_percentage=20,
        drop_probability=None,
        density=0.8,
        b_weight_alpha=None,
        variant_label=None,
        normalise_a_factors=None,
        scale_aware_ties=None,
        gpa_module_count=gpa_module_count,
        gpa_modules_hitting_max_iter=gpa_modules_hitting_max_iter,
        gpa_max_iter_fraction=gpa_fraction,
    )


class AnalyzeWeek3SweepTests(unittest.TestCase):
    def test_best_hparams_payload_selects_best_enhanced_variant(self) -> None:
        records = [
            make_record(method_key="task_arithmetic", avg=0.45, cola=0.0, source_path="task_arithmetic/lambda_1.json"),
            make_record(method_key="ties", avg=0.39, cola=0.0, source_path="ties/trim_10/lambda_0p7.json"),
            make_record(method_key="dare_ties", avg=0.40, cola=0.05, source_path="dare_ties/drop_0p5/trim_20/lambda_1.json"),
            make_record(method_key="lr_knots", avg=0.37, cola=0.0, source_path="lr_knots/trim_10/lambda_0p5.json"),
            make_record(method_key="gpa_baseline", avg=0.39, cola=0.0, source_path="gpa_ties/baseline/trim_20/lambda_0p2.json"),
            make_record(method_key="gpa_dgpa_ties", avg=0.37, cola=0.0, source_path="gpa_ties/dgpa_ties/trim_10/lambda_1.json"),
            make_record(method_key="gpa_dgpa_saties", avg=0.38, cola=0.0, source_path="gpa_ties/dgpa_saties/trim_10/lambda_0p15.json"),
            make_record(method_key="gpa_dgpa_saties_wb_0p5", avg=0.40, cola=0.0, source_path="gpa_ties/dgpa_saties_wb_0p5/trim_20/lambda_0p25.json"),
            make_record(method_key="gpa_dgpa_saties_wb_1p0", avg=0.385, cola=0.0, source_path="gpa_ties/dgpa_saties_wb_1p0/trim_20/lambda_0p2.json"),
        ]
        oracle = make_record(method_key="oracle", avg=0.70, cola=0.65, source_path="adapters/*/eval_metrics.json")

        payload = build_best_hparams_payload(records, oracle, source_root=PROJECT_ROOT / "results" / "hp_sweep_low_storage")

        self.assertEqual(
            payload["methods"]["gpa_best_enhanced"]["method_key"],
            "gpa_dgpa_saties_wb_0p5",
        )
        self.assertIn("oracle", payload["methods"])

    def test_gpa_rerun_decision_recommends_rerun_when_saturated_and_collapsed(self) -> None:
        records = [
            make_record(
                method_key="gpa_baseline",
                avg=0.39,
                cola=0.0,
                source_path="gpa_ties/baseline/trim_20/lambda_0p2.json",
                gpa_fraction=0.9,
            ),
            make_record(
                method_key="gpa_dgpa_saties_wb_0p5",
                avg=0.395,
                cola=0.0,
                source_path="gpa_ties/dgpa_saties_wb_0p5/trim_20/lambda_0p25.json",
                gpa_fraction=0.92,
            ),
        ]

        payload = build_gpa_rerun_decision_payload(records, source_root=PROJECT_ROOT / "results" / "hp_sweep_low_storage")

        self.assertTrue(payload["rerun_recommended"])
        self.assertGreater(payload["gpa_max_iter_saturation_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
