import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_adapters import parse_lora_key


class ValidateAdaptersTests(unittest.TestCase):
    def test_parse_lora_key_extracts_module_and_part(self) -> None:
        parsed = parse_lora_key("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
        self.assertEqual(parsed, ("base_model.model.model.layers.0.self_attn.q_proj", "lora_A"))

    def test_parse_lora_key_ignores_non_lora_tensors(self) -> None:
        self.assertIsNone(parse_lora_key("base_model.model.score.weight"))


if __name__ == "__main__":
    unittest.main()
