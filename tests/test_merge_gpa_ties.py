import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa_align_adapters import align_adapter_bundles
from scripts.merge_gpa_ties import (
    merge_gpa_ties_state_dicts,
    ties_merge_scale_aware,
    weighted_B_average,
)


class MergeGpaTiesTests(unittest.TestCase):
    def test_identical_adapters_round_trip_through_gpa_ties(self) -> None:
        module_name = "base_model.model.layers.0.self_attn.q_proj"
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        adapter_a = {
            "task": "task_a",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base", "r": 2, "rank_pattern": {}},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }
        adapter_b = {
            "task": "task_b",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base", "r": 2, "rank_pattern": {}},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }

        aligned_bundles, _ = align_adapter_bundles(
            [adapter_a, adapter_b],
            max_iter=20,
            tol=1e-12,
            init="first",
        )
        merged_state_dict, module_summaries = merge_gpa_ties_state_dicts(
            aligned_bundles,
            density=1.0,
            majority_sign_method="total",
            merge_weight=1.0,
        )

        torch.testing.assert_close(merged_state_dict[a_key], adapter_a["state_dict"][a_key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(merged_state_dict[b_key], adapter_a["state_dict"][b_key], atol=1e-5, rtol=1e-5)
        self.assertEqual(module_summaries[0]["module_name"], module_name)

    def test_scale_aware_ties_can_overrule_large_norm_sign_bias(self) -> None:
        merged, norms, target_norm = ties_merge_scale_aware(
            [
                torch.tensor([10.0], dtype=torch.float32),
                torch.tensor([-2.0], dtype=torch.float32),
                torch.tensor([-2.0], dtype=torch.float32),
            ],
            density=1.0,
            majority_sign_method="total",
        )

        self.assertEqual(norms, [10.0, 2.0, 2.0])
        self.assertAlmostEqual(target_norm, 14.0 / 3.0, places=6)
        self.assertLess(float(merged.item()), 0.0)

    def test_weighted_b_average_uses_inverse_norm_weights(self) -> None:
        merged, norms, weights = weighted_B_average(
            [
                torch.tensor([[4.0]], dtype=torch.float32),
                torch.tensor([[1.0]], dtype=torch.float32),
            ],
            alpha=1.0,
        )

        self.assertEqual(norms, [4.0, 1.0])
        self.assertAlmostEqual(weights[0], 0.2, places=6)
        self.assertAlmostEqual(weights[1], 0.8, places=6)
        torch.testing.assert_close(merged, torch.tensor([[1.6]], dtype=torch.float32))

    def test_enhanced_pipeline_still_round_trips_identical_adapters(self) -> None:
        module_name = "base_model.model.layers.0.self_attn.q_proj"
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        adapter_a = {
            "task": "task_a",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base", "r": 2, "rank_pattern": {}},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }
        adapter_b = {
            "task": "task_b",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base", "r": 2, "rank_pattern": {}},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }

        aligned_bundles, _ = align_adapter_bundles(
            [adapter_a, adapter_b],
            max_iter=20,
            tol=1e-12,
            init="first",
            normalise=True,
        )
        merged_state_dict, _ = merge_gpa_ties_state_dicts(
            aligned_bundles,
            density=1.0,
            majority_sign_method="total",
            merge_weight=1.0,
            scale_aware_ties=True,
            b_weight_alpha=1.0,
        )

        torch.testing.assert_close(merged_state_dict[a_key], adapter_a["state_dict"][a_key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(merged_state_dict[b_key], adapter_a["state_dict"][b_key], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
