import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import TASK_CONFIG, build_tokenize_function, build_tokenized_cache_file_names, get_task_config


class DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {"num_inputs": len(args), "kwargs": kwargs}


class DataModuleTests(unittest.TestCase):
    def test_task_config_contains_expected_tasks(self) -> None:
        self.assertEqual(set(TASK_CONFIG), {"sst2", "mnli", "qnli", "cola", "rte"})
        self.assertEqual(get_task_config("mnli").validation_split, "validation_matched")
        self.assertEqual(get_task_config("cola").metric_for_best_model, "matthews_correlation")

    def test_tokenize_function_handles_single_sentence_tasks(self) -> None:
        tokenize = build_tokenize_function(DummyTokenizer(), "sst2", max_length=128)
        result = tokenize({"sentence": ["hello world"]})
        self.assertEqual(result["num_inputs"], 1)
        self.assertEqual(result["kwargs"]["max_length"], 128)

    def test_tokenize_function_handles_sentence_pairs(self) -> None:
        tokenize = build_tokenize_function(DummyTokenizer(), "mnli", max_length=256)
        result = tokenize({"premise": ["p"], "hypothesis": ["h"]})
        self.assertEqual(result["num_inputs"], 2)
        self.assertEqual(result["kwargs"]["max_length"], 256)

    def test_build_tokenized_cache_file_names_is_split_stable(self) -> None:
        cache_files = build_tokenized_cache_file_names(
            {"train": object(), "validation": object()},
            task_name="qnli",
            max_length=256,
            tokenizer_id="Qwen/Qwen2.5-1.5B@256",
            cache_root=PROJECT_ROOT / ".tmp_test_cache",
        )
        self.assertIsNotNone(cache_files)
        self.assertIn("train", cache_files)
        self.assertIn("validation", cache_files)
        self.assertTrue(cache_files["train"].endswith("_train.arrow"))
        self.assertTrue(cache_files["validation"].endswith("_validation.arrow"))


if __name__ == "__main__":
    unittest.main()
