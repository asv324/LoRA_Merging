import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# `scripts.adapter_eval` imports `transformers` at module load time, which is not
# available in every contributor's local environment. Guard the import so the
# pure-python tests still run locally and the HF-dependent tests are skipped.
try:
    from scripts.adapter_eval import (  # noqa: E402
        predicted_labels_from_output,
        summarize_prediction_distribution,
    )

    _ADAPTER_EVAL_AVAILABLE = True
    _ADAPTER_EVAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - only exercised in bare environments
    _ADAPTER_EVAL_AVAILABLE = False
    _ADAPTER_EVAL_IMPORT_ERROR = exc


@unittest.skipUnless(_ADAPTER_EVAL_AVAILABLE, f"scripts.adapter_eval unavailable: {_ADAPTER_EVAL_IMPORT_ERROR!r}")
class AdapterEvalTests(unittest.TestCase):
    def test_predicted_labels_from_output_uses_argmax(self) -> None:
        predictions = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        labels = predicted_labels_from_output(predictions)
        np.testing.assert_array_equal(labels, np.array([1, 0, 1]))

    def test_summarize_prediction_distribution_flags_degenerate_predictions(self) -> None:
        predicted_labels = np.array([1] * 96 + [0] * 4, dtype=np.int64)
        summary = summarize_prediction_distribution(predicted_labels, num_labels=2)

        self.assertEqual(summary["class_counts"], [4, 96])
        self.assertEqual(summary["dominant_class"], 1)
        self.assertTrue(summary["degenerate_prediction"])


@unittest.skipUnless(_ADAPTER_EVAL_AVAILABLE, f"scripts.adapter_eval unavailable: {_ADAPTER_EVAL_IMPORT_ERROR!r}")
class RestoreClassifierHeadTests(unittest.TestCase):
    """Round-trip tests for `_restore_classifier_head` on a mock model.

    We avoid loading a real HF model (which would require downloading Qwen2.5)
    by constructing a tiny `nn.Module` with the same `.base_model.model.score`
    attribute path used by PEFT-wrapped `AutoModelForSequenceClassification`.
    """

    def _make_mock_model(self, hidden_size: int, num_labels: int):
        import torch.nn as nn  # imported lazily: test is skipped when torch is unavailable

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.score = nn.Linear(hidden_size, num_labels, bias=False)

        class _BaseWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = _Inner()

        class _PeftWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = _BaseWrapper()

        return _PeftWrapper()

    def test_restores_trained_head_when_present(self) -> None:
        import torch
        from safetensors.torch import save_file

        from scripts.adapter_eval import (
            CLASSIFIER_HEAD_SUBDIR,
            CLASSIFIER_HEAD_TENSOR_KEY,
            _restore_classifier_head,
        )

        hidden_size = 16
        num_labels = 3
        saved_head = torch.randn(num_labels, hidden_size, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = Path(tmp)
            heads_dir = adapter_path / CLASSIFIER_HEAD_SUBDIR
            heads_dir.mkdir(parents=True, exist_ok=True)
            save_file({CLASSIFIER_HEAD_TENSOR_KEY: saved_head}, str(heads_dir / "mnli.safetensors"))

            model = self._make_mock_model(hidden_size=hidden_size, num_labels=num_labels)
            pre_restoration = model.base_model.model.score.weight.detach().clone()

            result = _restore_classifier_head(
                model=model,
                adapter_path=adapter_path,
                task="mnli",
                num_labels=num_labels,
            )

        self.assertTrue(result["restored"])
        self.assertFalse(result["replaced_quantized_score"])
        restored_weight = model.base_model.model.score.weight.detach()
        torch.testing.assert_close(restored_weight.to(torch.float32), saved_head)
        self.assertFalse(torch.allclose(restored_weight, pre_restoration))

    def test_reports_missing_head_without_mutating_model(self) -> None:
        import torch

        from scripts.adapter_eval import _restore_classifier_head

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = Path(tmp)
            model = self._make_mock_model(hidden_size=8, num_labels=2)
            pre_restoration = model.base_model.model.score.weight.detach().clone()

            result = _restore_classifier_head(
                model=model,
                adapter_path=adapter_path,
                task="sst2",
                num_labels=2,
            )

        self.assertFalse(result["restored"])
        self.assertEqual(result["reason"], "no_classifier_head_file")
        torch.testing.assert_close(
            model.base_model.model.score.weight.detach(),
            pre_restoration,
        )

    def test_raises_on_shape_mismatch(self) -> None:
        import torch
        from safetensors.torch import save_file

        from scripts.adapter_eval import (
            CLASSIFIER_HEAD_SUBDIR,
            CLASSIFIER_HEAD_TENSOR_KEY,
            _restore_classifier_head,
        )

        hidden_size = 16
        num_labels = 2
        wrong_head = torch.randn(3, hidden_size, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            adapter_path = Path(tmp)
            heads_dir = adapter_path / CLASSIFIER_HEAD_SUBDIR
            heads_dir.mkdir(parents=True, exist_ok=True)
            save_file({CLASSIFIER_HEAD_TENSOR_KEY: wrong_head}, str(heads_dir / "sst2.safetensors"))

            model = self._make_mock_model(hidden_size=hidden_size, num_labels=num_labels)

            with self.assertRaisesRegex(ValueError, "Classifier head shape mismatch"):
                _restore_classifier_head(
                    model=model,
                    adapter_path=adapter_path,
                    task="sst2",
                    num_labels=num_labels,
                )


if __name__ == "__main__":
    unittest.main()
