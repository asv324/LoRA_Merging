"""Validate trained adapters and export LoRA shape/norm summaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASKS = ["sst2", "mnli", "qnli", "cola", "rte"]
TARGETS = {
    "sst2": ("eval_accuracy", 0.92, 0.95),
    "mnli": ("eval_accuracy", 0.82, 0.87),
    "qnli": ("eval_accuracy", 0.88, 0.92),
    "cola": ("eval_matthews_correlation", 0.50, 0.65),
    "rte": ("eval_accuracy", 0.70, 0.78),
}
ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
LORA_KEY_PATTERN = re.compile(r"^(?P<module>.+)\.(?P<lora_part>lora_[AB])\.weight$")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--configs-dir", "--configs_dir", dest="configs_dir", default=PROJECT_ROOT / "configs")
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", default=PROJECT_ROOT / "results")
    parser.add_argument("--rerun-eval", action="store_true")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    return parser


def parse_lora_key(key: str) -> Tuple[str, str] | None:
    match = LORA_KEY_PATTERN.match(key)
    if match is None:
        return None
    return match.group("module"), match.group("lora_part")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def load_metrics(task: str, adapter_dir: Path, rerun_eval: bool, eval_batch_size: int) -> Dict[str, object]:
    if rerun_eval:
        from scripts.adapter_eval import evaluate_adapter

        return to_jsonable(evaluate_adapter(task=task, adapter_dir=str(adapter_dir), batch_size=eval_batch_size))
    metrics_path = adapter_dir / "eval_metrics.json"
    return to_jsonable(json.loads(metrics_path.read_text(encoding="utf-8")))


def analyze_adapter_tensors(adapter_dir: Path) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    state_dict = load_file(str(adapter_dir / "adapter_model.safetensors"))
    grouped: Dict[str, Dict[str, object]] = defaultdict(dict)
    layer_entries: List[Dict[str, object]] = []

    for key, tensor in state_dict.items():
        parsed = parse_lora_key(key)
        if parsed is None:
            continue
        module_name, lora_part = parsed
        grouped[module_name][lora_part] = {
            "key": key,
            "shape": list(tensor.shape),
            "frobenius_norm": float(torch_norm(tensor)),
        }

    for module_name, values in sorted(grouped.items()):
        if "lora_A" not in values or "lora_B" not in values:
            continue
        module_type = module_name.split(".")[-1]
        a_shape = values["lora_A"]["shape"]
        b_shape = values["lora_B"]["shape"]
        layer_entries.append(
            {
                "module_name": module_name,
                "module_type": module_type,
                "group": "attention" if module_type in ATTENTION_MODULES else "mlp",
                "lora_A_key": values["lora_A"]["key"],
                "lora_B_key": values["lora_B"]["key"],
                "lora_A_shape": a_shape,
                "lora_B_shape": b_shape,
                "rank": int(a_shape[0]),
                "input_dimension": int(a_shape[1]),
                "output_dimension": int(b_shape[0]),
                "lora_A_frobenius_norm": values["lora_A"]["frobenius_norm"],
                "lora_B_frobenius_norm": values["lora_B"]["frobenius_norm"],
            }
        )

    summary = {
        "num_lora_layers": len(layer_entries),
        "attention_input_dimensions": sorted({entry["input_dimension"] for entry in layer_entries if entry["group"] == "attention"}),
        "attention_ranks": sorted({entry["rank"] for entry in layer_entries if entry["group"] == "attention"}),
        "all_attention_din_1536": all(entry["input_dimension"] == 1536 for entry in layer_entries if entry["group"] == "attention"),
        "all_ranks_16": all(entry["rank"] == 16 for entry in layer_entries),
    }
    return summary, layer_entries


def torch_norm(tensor) -> float:
    return float(tensor.float().norm(p="fro").item())


def average(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_norm_summary(layer_entries: List[Dict[str, object]]) -> Dict[str, float]:
    a_norms = [entry["lora_A_frobenius_norm"] for entry in layer_entries]
    b_norms = [entry["lora_B_frobenius_norm"] for entry in layer_entries]
    attention_a = [entry["lora_A_frobenius_norm"] for entry in layer_entries if entry["group"] == "attention"]
    attention_b = [entry["lora_B_frobenius_norm"] for entry in layer_entries if entry["group"] == "attention"]
    mlp_a = [entry["lora_A_frobenius_norm"] for entry in layer_entries if entry["group"] == "mlp"]
    mlp_b = [entry["lora_B_frobenius_norm"] for entry in layer_entries if entry["group"] == "mlp"]

    return {
        "avg_lora_A_norm": average(a_norms),
        "avg_lora_B_norm": average(b_norms),
        "avg_attention_lora_A_norm": average(attention_a),
        "avg_attention_lora_B_norm": average(attention_b),
        "avg_mlp_lora_A_norm": average(mlp_a),
        "avg_mlp_lora_B_norm": average(mlp_b),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    adapters_dir = Path(args.adapters_dir)
    configs_dir = Path(args.configs_dir)
    results_dir = Path(args.results_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    task_summaries = []
    canonical_mapping = None
    performance_checks = {}

    for task in TASKS:
        adapter_dir = adapters_dir / task
        metrics = load_metrics(task=task, adapter_dir=adapter_dir, rerun_eval=args.rerun_eval, eval_batch_size=args.eval_batch_size)
        mapping_summary, layer_entries = analyze_adapter_tensors(adapter_dir)
        norm_summary = build_norm_summary(layer_entries)

        target_metric, lower, upper = TARGETS[task]
        metric_value = float(metrics[target_metric])
        performance_checks[task] = {
            "metric_name": target_metric,
            "metric_value": metric_value,
            "target_lower": lower,
            "target_upper": upper,
            "within_target_range": lower <= metric_value <= upper,
        }

        task_summary = {
            "task": task,
            "adapter_dir": str(adapter_dir),
            "metrics": metrics,
            "performance_check": performance_checks[task],
            "shape_summary": mapping_summary,
            "norm_summary": norm_summary,
            "layer_norms": layer_entries,
        }
        task_summaries.append(task_summary)

        if canonical_mapping is None:
            canonical_mapping = {
                "base_task": task,
                "layers": [
                    {
                        "module_name": entry["module_name"],
                        "module_type": entry["module_type"],
                        "group": entry["group"],
                        "lora_A_key": entry["lora_A_key"],
                        "lora_B_key": entry["lora_B_key"],
                        "lora_A_shape": entry["lora_A_shape"],
                        "lora_B_shape": entry["lora_B_shape"],
                        "rank": entry["rank"],
                        "input_dimension": entry["input_dimension"],
                        "output_dimension": entry["output_dimension"],
                    }
                    for entry in layer_entries
                ],
            }

    norm_ranking = sorted(
        (
            {
                "task": summary["task"],
                "avg_lora_A_norm": summary["norm_summary"]["avg_lora_A_norm"],
                "avg_lora_B_norm": summary["norm_summary"]["avg_lora_B_norm"],
            }
            for summary in task_summaries
        ),
        key=lambda item: item["avg_lora_A_norm"] + item["avg_lora_B_norm"],
        reverse=True,
    )

    mapping_payload = {
        "model_name": "Qwen/Qwen2.5-1.5B",
        "tasks_verified": TASKS,
        "verification_summary": {
            "all_attention_din_1536": all(summary["shape_summary"]["all_attention_din_1536"] for summary in task_summaries),
            "all_ranks_16": all(summary["shape_summary"]["all_ranks_16"] for summary in task_summaries),
        },
        **(canonical_mapping or {"layers": []}),
    }

    norm_payload = {
        "experiment": "adapter_norm_analysis",
        "tasks": task_summaries,
        "performance_checks": performance_checks,
        "norm_ranking": norm_ranking,
    }

    (configs_dir / "lora_param_mapping.json").write_text(json.dumps(mapping_payload, indent=2), encoding="utf-8")
    (results_dir / "adapter_norm_analysis.json").write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")
    print("Saved configs/lora_param_mapping.json")
    print("Saved results/adapter_norm_analysis.json")


if __name__ == "__main__":
    main()
