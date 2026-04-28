import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_merged import (
    infer_stored_lambda,
    infer_trained_tasks,
    scale_lora_a_state_dict,
    summarize_evaluations,
    write_scaled_adapter_copy,
)


class EvaluateMergedTests(unittest.TestCase):
    def test_infer_trained_tasks_prefers_merge_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir)
            (adapter_dir / "merge_metadata.json").write_text(
                json.dumps({"source_tasks": ["cola", "rte"], "lambda": 0.5}),
                encoding="utf-8",
            )
            tasks = infer_trained_tasks(adapter_dir, explicit_tasks=None)
        self.assertEqual(tasks, ["cola", "rte"])

    def test_infer_stored_lambda_uses_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir)
            (adapter_dir / "merge_metadata.json").write_text(json.dumps({"lambda": 0.3}), encoding="utf-8")
            stored_lambda = infer_stored_lambda(adapter_dir, cli_value=None)
        self.assertEqual(stored_lambda, 0.3)

    def test_scale_lora_a_state_dict_only_scales_a_factors(self) -> None:
        state_dict = {
            "layer.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "layer.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            "other.weight": torch.tensor([5.0], dtype=torch.float32),
        }
        scaled = scale_lora_a_state_dict(state_dict, scale_factor=2.5)

        torch.testing.assert_close(scaled["layer.lora_A.weight"], torch.tensor([[2.5, 5.0]], dtype=torch.float32))
        torch.testing.assert_close(scaled["layer.lora_B.weight"], state_dict["layer.lora_B.weight"])
        torch.testing.assert_close(scaled["other.weight"], state_dict["other.weight"])

    def test_summarize_evaluations_uses_task_primary_metrics(self) -> None:
        evaluations = {
            "cola": {"eval_matthews_correlation": 0.4},
            "rte": {"eval_accuracy": 0.8},
        }
        summary = summarize_evaluations(evaluations)

        self.assertEqual(summary["primary_metrics"]["cola"]["metric"], "matthews_correlation")
        self.assertEqual(summary["primary_metrics"]["rte"]["metric"], "accuracy")
        self.assertAlmostEqual(summary["average_primary_score"], 0.6)

    def test_write_scaled_adapter_copy_updates_weights_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            output_dir = Path(temp_dir) / "scaled"
            source_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "layer.lora_A.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                    "layer.lora_B.weight": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
                },
                str(source_dir / "adapter_model.safetensors"),
            )
            (source_dir / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "dummy/base"}), encoding="utf-8")
            (source_dir / "merge_metadata.json").write_text(json.dumps({"lambda": 0.5}), encoding="utf-8")

            write_scaled_adapter_copy(source_dir, output_dir, target_lambda=1.0, stored_lambda=0.5)

            scaled_state = load_file(str(output_dir / "adapter_model.safetensors"))
            metadata = json.loads((output_dir / "merge_metadata.json").read_text(encoding="utf-8"))

        torch.testing.assert_close(scaled_state["layer.lora_A.weight"], torch.tensor([[2.0, 4.0]], dtype=torch.float32))
        torch.testing.assert_close(scaled_state["layer.lora_B.weight"], torch.tensor([[3.0], [4.0]], dtype=torch.float32))
        self.assertEqual(metadata["lambda"], 1.0)
        self.assertEqual(metadata["rescaled_from_lambda"], 0.5)


if __name__ == "__main__":
    unittest.main()
