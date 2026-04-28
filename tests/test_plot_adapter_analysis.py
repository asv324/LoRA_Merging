import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_adapter_analysis import build_group_matrix, build_layer_label, parse_layer_index


class PlotAdapterAnalysisTests(unittest.TestCase):
    def test_parse_layer_index_extracts_transformer_depth(self) -> None:
        value = parse_layer_index("base_model.model.model.layers.17.self_attn.q_proj")
        self.assertEqual(value, 17)

    def test_build_layer_label_shortens_module_name(self) -> None:
        label = build_layer_label("base_model.model.model.layers.3.self_attn.o_proj")
        self.assertEqual(label, "L03 attn.o_proj")

    def test_build_group_matrix_respects_canonical_module_order(self) -> None:
        group_layers = [
            {"module_name": "base_model.model.model.layers.0.self_attn.k_proj"},
            {"module_name": "base_model.model.model.layers.0.self_attn.q_proj"},
        ]
        task_lookup = {
            "sst2": {
                "layer_norms": [
                    {
                        "module_name": "base_model.model.model.layers.0.self_attn.q_proj",
                        "lora_A_frobenius_norm": 3.0,
                    },
                    {
                        "module_name": "base_model.model.model.layers.0.self_attn.k_proj",
                        "lora_A_frobenius_norm": 1.0,
                    },
                ]
            },
            "mnli": {
                "layer_norms": [
                    {
                        "module_name": "base_model.model.model.layers.0.self_attn.q_proj",
                        "lora_A_frobenius_norm": 4.0,
                    },
                    {
                        "module_name": "base_model.model.model.layers.0.self_attn.k_proj",
                        "lora_A_frobenius_norm": 2.0,
                    },
                ]
            },
        }

        matrix, labels = build_group_matrix(
            task_order=["sst2", "mnli"],
            group_layers=group_layers,
            task_lookup=task_lookup,
            field_name="lora_A_frobenius_norm",
        )

        np.testing.assert_allclose(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(labels, ["L00 attn.k_proj", "L00 attn.q_proj"])


if __name__ == "__main__":
    unittest.main()
