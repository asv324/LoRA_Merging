import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_task_arithmetic import (
    CLASSIFIER_HEAD_MANIFEST_FILENAME,
    CLASSIFIER_HEAD_OUTPUT_KEY,
    CLASSIFIER_HEAD_SOURCE_KEY,
    CLASSIFIER_HEAD_SUBDIR,
    build_merged_adapter_config,
    copy_classifier_heads,
    merge_task_arithmetic_state_dicts,
)


class MergeTaskArithmeticTests(unittest.TestCase):
    def test_exact_merge_matches_weighted_delta_sum(self) -> None:
        module_name = "base_model.model.layers.0.self_attn.q_proj"
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        adapter_a = {
            "task": "task_a",
            "adapter_dir": "unused",
            "config": {
                "r": 1,
                "lora_alpha": 2,
                "rank_pattern": {},
                "alpha_pattern": {},
                "base_model_name_or_path": "dummy/base",
            },
            "state_dict": {
                a_key: torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                b_key: torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }
        adapter_b = {
            "task": "task_b",
            "adapter_dir": "unused",
            "config": {
                "r": 1,
                "lora_alpha": 2,
                "rank_pattern": {},
                "alpha_pattern": {},
                "base_model_name_or_path": "dummy/base",
            },
            "state_dict": {
                a_key: torch.tensor([[5.0, -1.0]], dtype=torch.float32),
                b_key: torch.tensor([[2.0], [1.0]], dtype=torch.float32),
            },
            "modules": [module_name],
        }

        merged_state_dict, merged_rank_pattern, _ = merge_task_arithmetic_state_dicts(
            [adapter_a, adapter_b],
            merge_weight=0.5,
        )
        merged_config = build_merged_adapter_config(adapter_a["config"], merged_rank_pattern)

        self.assertEqual(merged_rank_pattern[module_name], 2)
        self.assertEqual(merged_config["r"], 2)
        self.assertEqual(merged_config["lora_alpha"], 2)

        merged_delta = merged_state_dict[b_key] @ merged_state_dict[a_key]
        expected_delta = 0.5 * (
            2.0 * (adapter_a["state_dict"][b_key] @ adapter_a["state_dict"][a_key])
            + 2.0 * (adapter_b["state_dict"][b_key] @ adapter_b["state_dict"][a_key])
        )
        torch.testing.assert_close(merged_delta, expected_delta)

    def test_merged_config_strips_modules_to_save(self) -> None:
        config = {
            "r": 16,
            "lora_alpha": 32,
            "modules_to_save": ["score", "classifier"],
            "rank_pattern": {},
            "alpha_pattern": {},
            "base_model_name_or_path": "dummy/base",
        }
        merged = build_merged_adapter_config(
            config,
            {"module_a": 80, "module_b": 80},
        )

        self.assertEqual(merged["r"], 80)
        self.assertEqual(merged["lora_alpha"], 80)
        self.assertIsNone(merged["modules_to_save"])
        self.assertEqual(merged["rank_pattern"], {})
        self.assertEqual(merged["alpha_pattern"], {})


class CopyClassifierHeadsTests(unittest.TestCase):
    def _make_bundle(self, task: str, head: torch.Tensor | None) -> dict:
        state_dict = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 2),
        }
        if head is not None:
            state_dict[CLASSIFIER_HEAD_SOURCE_KEY] = head
        return {
            "task": task,
            "adapter_dir": f"/fake/adapters/{task}",
            "config": {},
            "state_dict": state_dict,
            "modules": ["base_model.model.layers.0.self_attn.q_proj"],
        }

    def test_round_trip_preserves_per_task_heads(self) -> None:
        heads = {
            "sst2": torch.randn(2, 8, dtype=torch.float32),
            "mnli": torch.randn(3, 8, dtype=torch.float32),
            "rte": torch.randn(2, 8, dtype=torch.float32),
        }
        bundles = [self._make_bundle(task, head) for task, head in heads.items()]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = copy_classifier_heads(bundles, output_dir)

            for task, head in heads.items():
                head_path = output_dir / CLASSIFIER_HEAD_SUBDIR / f"{task}.safetensors"
                self.assertTrue(head_path.exists(), f"missing head file for {task}")
                restored = load_file(str(head_path))
                self.assertEqual(list(restored.keys()), [CLASSIFIER_HEAD_OUTPUT_KEY])
                torch.testing.assert_close(restored[CLASSIFIER_HEAD_OUTPUT_KEY], head)

            manifest_path = output_dir / CLASSIFIER_HEAD_MANIFEST_FILENAME
            self.assertTrue(manifest_path.exists())
            on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk_manifest, manifest)
            self.assertEqual(on_disk_manifest["schema_version"], 1)
            self.assertEqual(on_disk_manifest["source_key"], CLASSIFIER_HEAD_SOURCE_KEY)
            self.assertEqual(on_disk_manifest["output_key"], CLASSIFIER_HEAD_OUTPUT_KEY)
            manifest_tasks = [entry["task"] for entry in on_disk_manifest["heads"]]
            self.assertEqual(set(manifest_tasks), set(heads.keys()))
            for entry, task in zip(on_disk_manifest["heads"], manifest_tasks):
                self.assertEqual(entry["shape"], list(heads[task].shape))

    def test_skips_bundles_without_classifier_head(self) -> None:
        bundles = [
            self._make_bundle("sst2", torch.randn(2, 8)),
            self._make_bundle("lm_only", None),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            manifest = copy_classifier_heads(bundles, output_dir)

            self.assertTrue((output_dir / CLASSIFIER_HEAD_SUBDIR / "sst2.safetensors").exists())
            self.assertFalse((output_dir / CLASSIFIER_HEAD_SUBDIR / "lm_only.safetensors").exists())
            self.assertEqual([entry["task"] for entry in manifest["heads"]], ["sst2"])


if __name__ == "__main__":
    unittest.main()
