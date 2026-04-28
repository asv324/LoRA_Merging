"""Unit tests for scripts/merge_ta_aligned.py (Step 4.2 / Experiment 12)."""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa_align_adapters import align_adapter_bundles
from scripts.merge_ta_aligned import (
    build_variant_label,
    compute_per_adapter_weights,
    merge_ta_aligned_state_dicts,
)
from scripts.merge_task_arithmetic import (
    merge_task_arithmetic_state_dicts,
)


def _make_adapter(task: str, a_tensor: torch.Tensor, b_tensor: torch.Tensor) -> dict:
    module_name = "base_model.model.layers.0.self_attn.q_proj"
    a_key = f"{module_name}.lora_A.weight"
    b_key = f"{module_name}.lora_B.weight"
    return {
        "task": task,
        "adapter_dir": "unused",
        "config": {
            "r": int(a_tensor.shape[0]),
            "lora_alpha": int(a_tensor.shape[0]),
            "rank_pattern": {},
            "alpha_pattern": {},
            "base_model_name_or_path": "dummy/base",
        },
        "state_dict": {
            a_key: a_tensor.clone(),
            b_key: b_tensor.clone(),
        },
        "modules": [module_name],
    }


def _merged_delta(state_dict: dict, module_name: str) -> torch.Tensor:
    a_key = f"{module_name}.lora_A.weight"
    b_key = f"{module_name}.lora_B.weight"
    return state_dict[b_key].float() @ state_dict[a_key].float()


class MergeTaAlignedTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.module_name = "base_model.model.layers.0.self_attn.q_proj"
        self.adapter_a = _make_adapter(
            "task_a",
            torch.tensor([[1.0, 2.0, 0.5], [0.0, 1.0, -1.0]], dtype=torch.float32),
            torch.tensor([[1.0, 0.5], [0.5, 1.0], [2.0, -1.0]], dtype=torch.float32),
        )
        self.adapter_b = _make_adapter(
            "task_b",
            torch.tensor([[0.5, -1.0, 2.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
            torch.tensor([[2.0, 1.0], [-0.5, 0.5], [1.0, 0.0]], dtype=torch.float32),
        )

    def test_gpa_aligned_ta_matches_plain_task_arithmetic(self) -> None:
        """GPA alignment preserves each adapter's effective delta, so plain
        aligned-TA must reproduce unaligned Task Arithmetic exactly.
        """
        aligned_bundles, _ = align_adapter_bundles(
            [deepcopy(self.adapter_a), deepcopy(self.adapter_b)],
            max_iter=50,
            tol=1e-10,
            init="first",
            normalise=False,
        )

        aligned_merged, _, _ = merge_ta_aligned_state_dicts(
            aligned_bundles,
            merge_weight=0.75,
            b_weight_alpha=0.0,
        )
        plain_merged, _, _ = merge_task_arithmetic_state_dicts(
            [self.adapter_a, self.adapter_b],
            merge_weight=0.75,
        )

        torch.testing.assert_close(
            _merged_delta(aligned_merged, self.module_name),
            _merged_delta(plain_merged, self.module_name),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_unweighted_aligned_sum_equals_scaled_delta_sum(self) -> None:
        """Regardless of whether rotations are identity or not, an unweighted
        aligned-TA with merge_weight=lambda must reconstruct the plain-TA
        weighted sum of raw deltas.
        """
        aligned_bundles, _ = align_adapter_bundles(
            [deepcopy(self.adapter_a), deepcopy(self.adapter_b)],
            max_iter=50,
            tol=1e-10,
            init="first",
            normalise=True,
        )

        merged_state_dict, _, _ = merge_ta_aligned_state_dicts(
            aligned_bundles,
            merge_weight=1.0,
            b_weight_alpha=0.0,
        )
        a_key = f"{self.module_name}.lora_A.weight"
        b_key = f"{self.module_name}.lora_B.weight"
        # alpha / r == 1.0 for both adapters because we set lora_alpha = r,
        # so the expected aligned-TA delta is simply the sum of raw per-adapter
        # reconstructed deltas scaled by merge_weight.
        expected_delta = 1.0 * (
            self.adapter_a["state_dict"][b_key] @ self.adapter_a["state_dict"][a_key]
            + self.adapter_b["state_dict"][b_key] @ self.adapter_b["state_dict"][a_key]
        )
        torch.testing.assert_close(
            _merged_delta(merged_state_dict, self.module_name),
            expected_delta,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_inverse_norm_weights_average_to_one(self) -> None:
        """The weighting scheme must preserve overall TA scale: Sum w_i = N."""
        aligned_bundles, _ = align_adapter_bundles(
            [deepcopy(self.adapter_a), deepcopy(self.adapter_b)],
            max_iter=50,
            tol=1e-10,
            init="first",
            normalise=False,
        )
        a_key = f"{self.module_name}.lora_A.weight"
        b_key = f"{self.module_name}.lora_B.weight"

        aligned_b = [bundle["state_dict"][b_key] for bundle in aligned_bundles]
        norms, weights = compute_per_adapter_weights(aligned_b, alpha=0.5)

        self.assertEqual(len(weights), 2)
        self.assertAlmostEqual(sum(weights), 2.0, places=6)
        self.assertGreater(norms[0], 0.0)
        self.assertGreater(norms[1], 0.0)

        # Smaller-norm adapter must receive the larger weight under alpha > 0.
        smaller_index = 0 if norms[0] < norms[1] else 1
        larger_index = 1 - smaller_index
        self.assertGreater(weights[smaller_index], weights[larger_index])

    def test_enhanced_aligned_ta_differs_from_plain_ta(self) -> None:
        """dGPA alignment + inverse-norm B weighting must NOT reduce to plain TA,
        because the weighting breaks the delta-preserving invariance that GPA
        alone enjoys.
        """
        aligned_bundles, _ = align_adapter_bundles(
            [deepcopy(self.adapter_a), deepcopy(self.adapter_b)],
            max_iter=50,
            tol=1e-10,
            init="first",
            normalise=True,
        )

        enhanced_merged, _, _ = merge_ta_aligned_state_dicts(
            aligned_bundles,
            merge_weight=1.0,
            b_weight_alpha=0.5,
        )
        plain_merged, _, _ = merge_task_arithmetic_state_dicts(
            [self.adapter_a, self.adapter_b],
            merge_weight=1.0,
        )

        enhanced_delta = _merged_delta(enhanced_merged, self.module_name)
        plain_delta = _merged_delta(plain_merged, self.module_name)
        difference_norm = float((enhanced_delta - plain_delta).norm(p="fro").item())
        self.assertGreater(difference_norm, 1e-4)

    def test_variant_labels(self) -> None:
        self.assertEqual(
            build_variant_label(normalise_a_factors=False, b_weight_alpha=0.0),
            "GPA-aligned TA",
        )
        self.assertEqual(
            build_variant_label(normalise_a_factors=True, b_weight_alpha=0.5),
            "dGPA-aligned TA + wB(0.5)",
        )


if __name__ == "__main__":
    unittest.main()
