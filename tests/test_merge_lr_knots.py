import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_lr_knots import decompose_stacked_a_factors, merge_lr_knots_state_dicts


class MergeLrKnotsTests(unittest.TestCase):
    def test_decompose_stacked_a_factors_reconstructs_inputs(self) -> None:
        a_matrices = [
            torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, -1.0], [1.0, 0.5]], dtype=torch.float32),
        ]

        latent_blocks, shared_basis, _ = decompose_stacked_a_factors(a_matrices)

        for latent, original in zip(latent_blocks, a_matrices):
            reconstructed = latent @ shared_basis
            torch.testing.assert_close(reconstructed, original, atol=1e-5, rtol=1e-5)

    def test_identical_adapters_round_trip_through_lr_knots(self) -> None:
        module_name = "base_model.model.layers.0.self_attn.q_proj"
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        adapter_a = {
            "task": "task_a",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base"},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }
        adapter_b = {
            "task": "task_b",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base"},
            "state_dict": {
                a_key: torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                b_key: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }

        merged_state_dict, module_summaries = merge_lr_knots_state_dicts(
            [adapter_a, adapter_b],
            density=1.0,
            majority_sign_method="total",
            merge_weight=1.0,
        )

        torch.testing.assert_close(merged_state_dict[a_key], adapter_a["state_dict"][a_key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(merged_state_dict[b_key], adapter_a["state_dict"][b_key])
        self.assertLess(module_summaries[0]["max_reconstruction_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
