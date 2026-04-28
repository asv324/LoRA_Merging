import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_ties import merge_ties_state_dicts, ties_merge_tensors


class MergeTiesTests(unittest.TestCase):
    def test_ties_merge_tensors_matches_manual_example(self) -> None:
        tensors = [
            torch.tensor([1.0, -0.5, 0.2, -3.0]),
            torch.tensor([2.0, 0.4, -0.1, 4.0]),
            torch.tensor([-0.5, -0.6, 0.3, 5.0]),
        ]
        weights = torch.tensor([1.0, 2.0, 1.0])

        merged = ties_merge_tensors(
            tensors,
            weights=weights,
            density=0.5,
            majority_sign_method="total",
        )

        expected = torch.tensor([2.5, -0.6, 0.0, 6.5])
        torch.testing.assert_close(merged, expected)

    def test_ties_merge_tensors_uses_weighted_sign_election(self) -> None:
        tensors = [
            torch.tensor([1.0]),
            torch.tensor([-2.0]),
        ]
        weights = torch.tensor([3.0, 1.0])

        merged = ties_merge_tensors(
            tensors,
            weights=weights,
            density=1.0,
            majority_sign_method="total",
        )

        torch.testing.assert_close(merged, torch.tensor([3.0]))

    def test_merge_ties_state_dicts_applies_lambda_after_merge(self) -> None:
        module_name = "base_model.model.layers.0.self_attn.q_proj"
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        adapter_a = {
            "task": "task_a",
            "adapter_dir": "unused",
            "config": {
                "r": 1,
                "lora_alpha": 1,
                "rank_pattern": {},
                "alpha_pattern": {},
                "base_model_name_or_path": "dummy/base",
            },
            "state_dict": {
                a_key: torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                b_key: torch.tensor([[2.0], [1.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }
        adapter_b = {
            "task": "task_b",
            "adapter_dir": "unused",
            "config": {
                "r": 1,
                "lora_alpha": 1,
                "rank_pattern": {},
                "alpha_pattern": {},
                "base_model_name_or_path": "dummy/base",
            },
            "state_dict": {
                a_key: torch.tensor([[3.0, 4.0]], dtype=torch.float32),
                b_key: torch.tensor([[4.0], [3.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }

        merged_state_dict, module_summaries = merge_ties_state_dicts(
            [adapter_a, adapter_b],
            density=1.0,
            majority_sign_method="total",
            merge_weight=0.5,
        )

        torch.testing.assert_close(merged_state_dict[a_key], torch.tensor([[1.0, 1.5]]))
        torch.testing.assert_close(merged_state_dict[b_key], torch.tensor([[3.0], [2.0]]))
        self.assertEqual(module_summaries[0]["module_name"], module_name)


if __name__ == "__main__":
    unittest.main()
