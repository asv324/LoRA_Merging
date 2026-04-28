import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_dare import apply_dare_to_adapter_bundles, apply_dare_to_tensor


class MergeDareTests(unittest.TestCase):
    def test_apply_dare_to_tensor_is_reproducible_and_rescaled(self) -> None:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        generator_a = torch.Generator(device="cpu")
        generator_a.manual_seed(7)
        first = apply_dare_to_tensor(tensor, drop_probability=0.5, generator=generator_a)

        generator_b = torch.Generator(device="cpu")
        generator_b.manual_seed(7)
        second = apply_dare_to_tensor(tensor, drop_probability=0.5, generator=generator_b)

        torch.testing.assert_close(first, second)
        nonzero_mask = first != 0
        torch.testing.assert_close(first[nonzero_mask], tensor[nonzero_mask] * 2.0)

    def test_apply_dare_to_adapter_bundles_is_identity_when_p_zero(self) -> None:
        bundle = {
            "task": "dummy",
            "adapter_dir": "unused",
            "config": {"base_model_name_or_path": "dummy/base"},
            "state_dict": OrderedDict(
                {
                    "module.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "module.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
                    "module.other.weight": torch.tensor([5.0], dtype=torch.float32),
                }
            ),
            "modules": ["module"],
        }

        sparsified = apply_dare_to_adapter_bundles([bundle], drop_probability=0.0, seed=123)[0]

        for key, tensor in bundle["state_dict"].items():
            torch.testing.assert_close(sparsified["state_dict"][key], tensor)


if __name__ == "__main__":
    unittest.main()
