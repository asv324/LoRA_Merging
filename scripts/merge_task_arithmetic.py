"""Task Arithmetic baseline for LoRA adapters.

This script merges LoRA adapters in effective delta-weight space:

    DeltaW_merge = lambda * sum_i DeltaW_i
    DeltaW_i = (alpha_i / r_i) * B_i @ A_i

For LoRA, naively summing A and B factors separately is not correct because it
introduces cross-terms. Instead, this implementation stores the exact weighted
sum by concatenating LoRA factors per module:

    B_merge = [B_1, ..., B_N]
    A_merge = [lambda * scale_1 * A_1;
               ...
               lambda * scale_N * A_N]

The merged adapter uses alpha = r, so its runtime scaling is 1 and the saved
factors reproduce the intended delta exactly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASKS = ["sst2", "mnli", "qnli", "cola", "rte"]
TOKENIZER_ASSETS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
]

# Per-task trained classifier heads stored alongside the merged adapter.
# The source GLUE adapters were trained with `modules_to_save=["score", "classifier"]`,
# so each adapter's `adapter_model.safetensors` contains a fully trained
# `base_model.model.score.weight`. Factor-space merges do not preserve this head
# (MNLI has `num_labels=3`, every other task has `num_labels=2`, so there is
# no single "merged head"), which previously forced `AutoModelForSequenceClassification`
# to re-initialise `score.weight` randomly at eval time. We now copy each source
# adapter's trained head into `<merged_adapter>/classifier_heads/<task>.safetensors`
# under the key `CLASSIFIER_HEAD_OUTPUT_KEY`, and `scripts/adapter_eval.py`
# restores the matching head onto the loaded model before running `Trainer.predict`.
CLASSIFIER_HEAD_SUBDIR = "classifier_heads"
CLASSIFIER_HEAD_SOURCE_KEY = "base_model.model.score.weight"
CLASSIFIER_HEAD_OUTPUT_KEY = "score.weight"
CLASSIFIER_HEAD_MANIFEST_FILENAME = "classifier_heads_manifest.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        help="Adapter task directories to merge. Default: the five Week 1 GLUE adapters.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values for the task-arithmetic sweep.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "merged_adapters" / "task_arithmetic",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "task_arithmetic_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation and only write merged adapters.")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    return parser


def parse_lambda_values(raw_value: str) -> List[float]:
    values = []
    for piece in raw_value.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0:
            raise ValueError("lambda values must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("at least one lambda value is required")
    return values


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


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def list_lora_modules(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    modules = []
    for key in state_dict:
        if key.endswith(".lora_A.weight"):
            modules.append(key[: -len(".lora_A.weight")])
    return sorted(modules)


def resolve_module_rank(config: Dict[str, object], module_name: str) -> int:
    rank_pattern = config.get("rank_pattern") or {}
    return int(rank_pattern.get(module_name, config["r"]))


def resolve_module_alpha(config: Dict[str, object], module_name: str) -> float:
    alpha_pattern = config.get("alpha_pattern") or {}
    return float(alpha_pattern.get(module_name, config["lora_alpha"]))


def resolve_module_scale(config: Dict[str, object], module_name: str) -> float:
    rank = resolve_module_rank(config, module_name)
    if rank <= 0:
        raise ValueError(f"rank must be positive for module {module_name}")
    return resolve_module_alpha(config, module_name) / rank


def format_lambda(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_adapter_bundle(adapter_dir: Path) -> Dict[str, object]:
    state_dict = load_file(str(adapter_dir / "adapter_model.safetensors"))
    config = read_json(adapter_dir / "adapter_config.json")
    modules = list_lora_modules(state_dict)
    if not modules:
        raise ValueError(f"No LoRA modules found in {adapter_dir}")

    return {
        "task": adapter_dir.name,
        "adapter_dir": str(adapter_dir),
        "config": config,
        "state_dict": state_dict,
        "modules": modules,
    }


def validate_compatible_adapters(adapter_bundles: Sequence[Dict[str, object]]) -> List[str]:
    if not adapter_bundles:
        raise ValueError("at least one adapter is required")

    reference_modules = adapter_bundles[0]["modules"]
    reference_base_model = adapter_bundles[0]["config"]["base_model_name_or_path"]

    for bundle in adapter_bundles[1:]:
        if bundle["modules"] != reference_modules:
            raise ValueError("all adapters must expose the same ordered LoRA modules")
        if bundle["config"]["base_model_name_or_path"] != reference_base_model:
            raise ValueError("all adapters must share the same base model")

    return list(reference_modules)


def merge_task_arithmetic_state_dicts(
    adapter_bundles: Sequence[Dict[str, object]],
    merge_weight: float,
) -> Tuple[OrderedDict[str, torch.Tensor], Dict[str, int], List[Dict[str, object]]]:
    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    merged_rank_pattern: Dict[str, int] = {}
    module_summaries: List[Dict[str, object]] = []

    modules = validate_compatible_adapters(adapter_bundles)
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        weighted_a_blocks = []
        b_blocks = []
        source_ranks = []
        reference_a_dtype = None
        reference_b_dtype = None
        reference_device = None

        for bundle in adapter_bundles:
            state_dict = bundle["state_dict"]
            config = bundle["config"]
            lora_a = state_dict[a_key]
            lora_b = state_dict[b_key]
            if reference_a_dtype is None:
                reference_a_dtype = lora_a.dtype
                reference_b_dtype = lora_b.dtype
                reference_device = lora_a.device

            scale = merge_weight * resolve_module_scale(config, module_name)
            weighted_a_blocks.append(lora_a.to(dtype=torch.float32) * scale)
            b_blocks.append(lora_b.to(dtype=torch.float32))
            source_ranks.append(int(lora_a.shape[0]))

        merged_rank = int(sum(source_ranks))
        merged_rank_pattern[module_name] = merged_rank

        merged_lora_a = torch.cat(weighted_a_blocks, dim=0).to(dtype=reference_a_dtype, device=reference_device)
        merged_lora_b = torch.cat(b_blocks, dim=1).to(dtype=reference_b_dtype, device=reference_device)
        merged_state_dict[a_key] = merged_lora_a.contiguous()
        merged_state_dict[b_key] = merged_lora_b.contiguous()

        merged_delta_norm = float((merged_lora_b.float() @ merged_lora_a.float()).norm(p="fro").item())
        module_summaries.append(
            {
                "module_name": module_name,
                "source_ranks": source_ranks,
                "merged_rank": merged_rank,
                "merged_delta_frobenius_norm": merged_delta_norm,
            }
        )

    return merged_state_dict, merged_rank_pattern, module_summaries


def build_merged_adapter_config(reference_config: Dict[str, object], merged_rank_pattern: Dict[str, int]) -> Dict[str, object]:
    merged_config = json.loads(json.dumps(reference_config))
    unique_ranks = sorted(set(merged_rank_pattern.values()))
    base_rank = unique_ranks[0]

    merged_config["r"] = base_rank
    merged_config["lora_alpha"] = base_rank
    merged_config["rank_pattern"] = {
        module_name: rank for module_name, rank in sorted(merged_rank_pattern.items()) if rank != base_rank
    }
    merged_config["alpha_pattern"] = {
        module_name: rank for module_name, rank in sorted(merged_rank_pattern.items()) if rank != base_rank
    }
    merged_config["modules_to_save"] = None
    merged_config["inference_mode"] = True
    return merged_config


def write_merged_readme(output_dir: Path, tasks: Sequence[str], merge_weight: float, merged_rank_pattern: Dict[str, int]) -> None:
    unique_ranks = sorted(set(merged_rank_pattern.values()))
    readme = (
        "---\n"
        "library_name: peft\n"
        "---\n\n"
        "# Task Arithmetic Merged Adapter\n\n"
        f"Merged source adapters: {', '.join(tasks)}\n\n"
        f"Lambda: {merge_weight}\n\n"
        "Method: exact task arithmetic in LoRA delta-weight space via factor concatenation.\n\n"
        f"Merged ranks: {', '.join(str(rank) for rank in unique_ranks)}\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def copy_tokenizer_assets(reference_adapter_dir: Path, output_dir: Path) -> None:
    for asset_name in TOKENIZER_ASSETS:
        source_path = reference_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)


def copy_classifier_heads(
    adapter_bundles: Sequence[Dict[str, object]],
    output_dir: Path,
) -> Dict[str, object]:
    """Copy every source adapter's trained classifier head into the merged adapter.

    For each bundle with a `base_model.model.score.weight` tensor in its state dict,
    writes `<output_dir>/classifier_heads/<task>.safetensors` containing a single key
    (`CLASSIFIER_HEAD_OUTPUT_KEY`). Also emits a `classifier_heads_manifest.json`
    listing every head copied so `scripts/adapter_eval.py` and downstream tooling
    can discover them without scanning the directory.

    Bundles without a classifier head (e.g. hypothetical LM-only adapters) are
    skipped silently so the helper can be called unconditionally from every merge
    script without forcing the head to exist.
    """
    heads_dir = output_dir / CLASSIFIER_HEAD_SUBDIR
    heads_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, object]] = []
    for bundle in adapter_bundles:
        task = str(bundle["task"])
        state_dict = bundle["state_dict"]
        head_tensor = state_dict.get(CLASSIFIER_HEAD_SOURCE_KEY)
        if head_tensor is None:
            continue
        head_path = heads_dir / f"{task}.safetensors"
        save_file(
            {CLASSIFIER_HEAD_OUTPUT_KEY: head_tensor.contiguous()},
            str(head_path),
        )
        entries.append(
            {
                "task": task,
                "source_adapter_dir": str(bundle["adapter_dir"]),
                "path": str(head_path.relative_to(output_dir)).replace("\\", "/"),
                "shape": list(head_tensor.shape),
                "dtype": str(head_tensor.dtype),
            }
        )

    manifest = {
        "schema_version": 1,
        "source_key": CLASSIFIER_HEAD_SOURCE_KEY,
        "output_key": CLASSIFIER_HEAD_OUTPUT_KEY,
        "heads": entries,
    }
    manifest_path = output_dir / CLASSIFIER_HEAD_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def save_merged_adapter(
    output_dir: Path,
    merged_state_dict: OrderedDict[str, torch.Tensor],
    merged_config: Dict[str, object],
    reference_adapter_dir: Path,
    tasks: Sequence[str],
    merge_weight: float,
    module_summaries: Sequence[Dict[str, object]],
    adapter_bundles: Sequence[Dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(merged_state_dict), str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(json.dumps(merged_config, indent=2), encoding="utf-8")
    write_merged_readme(output_dir, tasks=tasks, merge_weight=merge_weight, merged_rank_pattern=module_rank_pattern_from_summaries(module_summaries))
    copy_tokenizer_assets(reference_adapter_dir, output_dir)
    if adapter_bundles is not None:
        copy_classifier_heads(adapter_bundles, output_dir)

    metadata = {
        "method": "task_arithmetic_exact_delta_space",
        "formula": "DeltaW_merge = lambda * sum_i DeltaW_i",
        "why_not_linear_factor_sum": "Adding LoRA A and B factors separately introduces cross-terms, so this adapter stores the exact weighted delta via concatenated factors.",
        "source_tasks": list(tasks),
        "lambda": merge_weight,
        "module_summaries": list(module_summaries),
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def module_rank_pattern_from_summaries(module_summaries: Sequence[Dict[str, object]]) -> Dict[str, int]:
    return {summary["module_name"]: int(summary["merged_rank"]) for summary in module_summaries}


def evaluate_adapter_on_tasks(
    adapter_dir: Path,
    tasks: Sequence[str],
    batch_size: int,
    max_eval_samples: int | None,
) -> Dict[str, object]:
    from scripts.adapter_eval import evaluate_adapter

    evaluations = {}
    for task in tasks:
        evaluations[task] = to_jsonable(
            evaluate_adapter(
                task=task,
                adapter_dir=str(adapter_dir),
                batch_size=batch_size,
                max_eval_samples=max_eval_samples,
            )
        )
    return evaluations


def main() -> None:
    args = build_arg_parser().parse_args()
    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    lambda_values = parse_lambda_values(args.lambdas)
    tasks = list(args.tasks)

    adapter_bundles = []
    for task in tasks:
        adapter_dir = adapters_dir / task
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        adapter_bundles.append(load_adapter_bundle(adapter_dir))

    validate_compatible_adapters(adapter_bundles)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "task_arithmetic_exact_delta_space",
        "source_tasks": tasks,
        "lambda_values": lambda_values,
        "adapters_dir": str(adapters_dir),
        "runs": [],
    }

    for merge_weight in lambda_values:
        merged_state_dict, merged_rank_pattern, module_summaries = merge_task_arithmetic_state_dicts(
            adapter_bundles,
            merge_weight=merge_weight,
        )
        merged_config = build_merged_adapter_config(adapter_bundles[0]["config"], merged_rank_pattern)
        merged_adapter_dir = output_dir / f"lambda_{format_lambda(merge_weight)}"

        save_merged_adapter(
            output_dir=merged_adapter_dir,
            merged_state_dict=merged_state_dict,
            merged_config=merged_config,
            reference_adapter_dir=Path(adapter_bundles[0]["adapter_dir"]),
            tasks=tasks,
            merge_weight=merge_weight,
            module_summaries=module_summaries,
            adapter_bundles=adapter_bundles,
        )

        run_summary = {
            "lambda": merge_weight,
            "adapter_dir": str(merged_adapter_dir),
            "merged_rank_pattern": merged_rank_pattern,
            "module_summaries": module_summaries,
        }
        if not args.skip_eval:
            run_summary["evaluation"] = evaluate_adapter_on_tasks(
                adapter_dir=merged_adapter_dir,
                tasks=tasks,
                batch_size=args.eval_batch_size,
                max_eval_samples=args.max_eval_samples,
            )
        results_payload["runs"].append(run_summary)
        results_path.write_text(json.dumps(to_jsonable(results_payload), indent=2), encoding="utf-8")
        print(f"Wrote merged adapter for lambda={merge_weight} to {merged_adapter_dir}")

    print(f"Saved task arithmetic summary to {results_path}")


if __name__ == "__main__":
    main()
