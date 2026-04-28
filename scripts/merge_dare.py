"""DARE baseline for LoRA adapter merging.

This script first sparsifies each adapter by randomly dropping parameters with
probability `p` and rescaling the surviving parameters by `1 / (1 - p)`. The
resulting sparsified adapters are then merged with either:

- exact Task Arithmetic in delta space, or
- factor-space TIES on raw LoRA factors.

The DARE step is applied independently to every stored `lora_A` and `lora_B`
tensor of each adapter.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_task_arithmetic import (
    TASKS,
    build_merged_adapter_config,
    copy_classifier_heads,
    evaluate_adapter_on_tasks,
    format_lambda,
    load_adapter_bundle,
    merge_task_arithmetic_state_dicts,
    parse_lambda_values,
    validate_compatible_adapters,
)
from scripts.merge_ties import merge_ties_state_dicts, parse_trim_percentages, rank_pattern_from_reference

TOKENIZER_ASSETS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--merge-methods",
        "--merge_methods",
        dest="merge_methods",
        nargs="+",
        choices=["task_arithmetic", "ties"],
        default=["task_arithmetic", "ties"],
        help="Merge methods to apply after DARE sparsification.",
    )
    parser.add_argument(
        "--drop-probabilities",
        "--drop_probabilities",
        dest="drop_probabilities",
        default="0.0,0.1,0.5,0.9",
        help="Comma-separated DARE drop probabilities p.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values applied after merging.",
    )
    parser.add_argument(
        "--trim-percentages",
        "--trim_percentages",
        dest="trim_percentages",
        default="10,20,30",
        help="Trim percentages used only when merge method includes TIES.",
    )
    parser.add_argument(
        "--majority-sign-method",
        "--majority_sign_method",
        dest="majority_sign_method",
        choices=["total", "frequency"],
        default="total",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DARE masks.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "merged_adapters" / "dare",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "dare_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    return parser


def parse_drop_probabilities(raw_value: str) -> List[float]:
    values = parse_lambda_values(raw_value)
    for value in values:
        if value < 0.0 or value >= 1.0:
            raise ValueError("drop probabilities must lie in [0, 1)")
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


def apply_dare_to_tensor(
    tensor: torch.Tensor,
    *,
    drop_probability: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if drop_probability == 0.0:
        return tensor.clone()

    keep_probability = 1.0 - drop_probability
    mask = torch.rand(tensor.shape, generator=generator, device=tensor.device) < keep_probability
    scaled = tensor.float() * mask.to(dtype=torch.float32) / keep_probability
    return scaled.to(dtype=tensor.dtype)


def apply_dare_to_adapter_bundles(
    adapter_bundles: Sequence[Dict[str, object]],
    *,
    drop_probability: float,
    seed: int,
) -> List[Dict[str, object]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sparsified_bundles: List[Dict[str, object]] = []

    for bundle in adapter_bundles:
        sparsified_state_dict = OrderedDict()
        for key, tensor in bundle["state_dict"].items():
            if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"):
                sparsified_state_dict[key] = apply_dare_to_tensor(
                    tensor,
                    drop_probability=drop_probability,
                    generator=generator,
                )
            else:
                sparsified_state_dict[key] = tensor.clone()

        sparsified_bundle = {
            "task": bundle["task"],
            "adapter_dir": bundle["adapter_dir"],
            "config": copy.deepcopy(bundle["config"]),
            "state_dict": sparsified_state_dict,
            "modules": list(bundle["modules"]),
        }
        sparsified_bundles.append(sparsified_bundle)

    return sparsified_bundles


def copy_tokenizer_assets(reference_adapter_dir: Path, output_dir: Path) -> None:
    for asset_name in TOKENIZER_ASSETS:
        source_path = reference_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)


def count_nonzero_fraction(state_dict: Dict[str, torch.Tensor], suffix: str) -> float:
    fractions = []
    for key, tensor in state_dict.items():
        if key.endswith(suffix):
            fractions.append(float((tensor != 0).float().mean().item()))
    return float(sum(fractions) / len(fractions)) if fractions else 0.0


def summarize_sparsified_bundles(adapter_bundles: Sequence[Dict[str, object]]) -> Dict[str, float]:
    a_fractions = [count_nonzero_fraction(bundle["state_dict"], ".lora_A.weight") for bundle in adapter_bundles]
    b_fractions = [count_nonzero_fraction(bundle["state_dict"], ".lora_B.weight") for bundle in adapter_bundles]
    return {
        "avg_nonzero_fraction_lora_A": float(sum(a_fractions) / len(a_fractions)) if a_fractions else 0.0,
        "avg_nonzero_fraction_lora_B": float(sum(b_fractions) / len(b_fractions)) if b_fractions else 0.0,
    }


def write_merged_adapter(
    *,
    output_dir: Path,
    merged_state_dict: OrderedDict[str, torch.Tensor],
    merged_config: Dict[str, object],
    reference_adapter_dir: Path,
    metadata: Dict[str, object],
    adapter_bundles: Sequence[Dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(merged_state_dict), str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(json.dumps(merged_config, indent=2), encoding="utf-8")
    (output_dir / "merge_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    copy_tokenizer_assets(reference_adapter_dir, output_dir)
    if adapter_bundles is not None:
        copy_classifier_heads(adapter_bundles, output_dir)

    readme = (
        "---\n"
        "library_name: peft\n"
        "---\n\n"
        "# DARE Merged Adapter\n\n"
        f"Merge method: {metadata['merge_method']}\n\n"
        f"Source adapters: {', '.join(metadata['source_tasks'])}\n\n"
        f"Drop probability: {metadata['drop_probability']}\n\n"
        f"Lambda: {metadata['lambda']}\n"
    )
    if "trim_percentage" in metadata:
        readme += f"\nTrim percentage: {metadata['trim_percentage']}\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    tasks = list(args.tasks)
    drop_probabilities = parse_drop_probabilities(args.drop_probabilities)
    lambda_values = parse_lambda_values(args.lambdas)
    trim_percentages = parse_trim_percentages(args.trim_percentages)

    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    validate_compatible_adapters(adapter_bundles)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "dare_adapter_sparsification",
        "source_tasks": tasks,
        "merge_methods": list(args.merge_methods),
        "drop_probabilities": drop_probabilities,
        "lambda_values": lambda_values,
        "trim_percentages": trim_percentages if "ties" in args.merge_methods else [],
        "majority_sign_method": args.majority_sign_method,
        "seed": args.seed,
        "runs": [],
    }

    for drop_probability in drop_probabilities:
        sparsified_bundles = apply_dare_to_adapter_bundles(
            adapter_bundles,
            drop_probability=drop_probability,
            seed=args.seed,
        )
        sparsified_summary = summarize_sparsified_bundles(sparsified_bundles)

        if "task_arithmetic" in args.merge_methods:
            for merge_weight in lambda_values:
                merged_state_dict, merged_rank_pattern, module_summaries = merge_task_arithmetic_state_dicts(
                    sparsified_bundles,
                    merge_weight=merge_weight,
                )
                merged_config = build_merged_adapter_config(sparsified_bundles[0]["config"], merged_rank_pattern)
                merged_adapter_dir = output_dir / "task_arithmetic" / f"drop_{format_lambda(drop_probability)}" / f"lambda_{format_lambda(merge_weight)}"
                metadata = {
                    "method": "dare_then_task_arithmetic",
                    "merge_method": "task_arithmetic",
                    "source_tasks": tasks,
                    "drop_probability": drop_probability,
                    "rescale_factor": 1.0 / (1.0 - drop_probability),
                    "lambda": merge_weight,
                    "seed": args.seed,
                    "sparsified_summary": sparsified_summary,
                    "module_summaries": module_summaries,
                }
                write_merged_adapter(
                    output_dir=merged_adapter_dir,
                    merged_state_dict=merged_state_dict,
                    merged_config=merged_config,
                    reference_adapter_dir=Path(sparsified_bundles[0]["adapter_dir"]),
                    metadata=metadata,
                    adapter_bundles=sparsified_bundles,
                )
                run_summary = {
                    "merge_method": "task_arithmetic",
                    "drop_probability": drop_probability,
                    "lambda": merge_weight,
                    "adapter_dir": str(merged_adapter_dir),
                    "sparsified_summary": sparsified_summary,
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
                print(f"Wrote DARE+TaskArithmetic adapter for p={drop_probability} lambda={merge_weight} to {merged_adapter_dir}")

        if "ties" in args.merge_methods:
            merged_rank_pattern = rank_pattern_from_reference(sparsified_bundles)
            merged_config = build_merged_adapter_config(sparsified_bundles[0]["config"], merged_rank_pattern)
            for trim_percentage in trim_percentages:
                density = 1.0 - (trim_percentage / 100.0)
                for merge_weight in lambda_values:
                    merged_state_dict, module_summaries = merge_ties_state_dicts(
                        sparsified_bundles,
                        density=density,
                        majority_sign_method=args.majority_sign_method,
                        merge_weight=merge_weight,
                    )
                    merged_adapter_dir = output_dir / "ties" / f"drop_{format_lambda(drop_probability)}" / f"trim_{trim_percentage}" / f"lambda_{format_lambda(merge_weight)}"
                    metadata = {
                        "method": "dare_then_ties",
                        "merge_method": "ties",
                        "source_tasks": tasks,
                        "drop_probability": drop_probability,
                        "rescale_factor": 1.0 / (1.0 - drop_probability),
                        "trim_percentage": trim_percentage,
                        "density": density,
                        "lambda": merge_weight,
                        "seed": args.seed,
                        "majority_sign_method": args.majority_sign_method,
                        "sparsified_summary": sparsified_summary,
                        "module_summaries": module_summaries,
                    }
                    write_merged_adapter(
                        output_dir=merged_adapter_dir,
                        merged_state_dict=merged_state_dict,
                        merged_config=merged_config,
                        reference_adapter_dir=Path(sparsified_bundles[0]["adapter_dir"]),
                        metadata=metadata,
                        adapter_bundles=sparsified_bundles,
                    )
                    run_summary = {
                        "merge_method": "ties",
                        "drop_probability": drop_probability,
                        "trim_percentage": trim_percentage,
                        "density": density,
                        "lambda": merge_weight,
                        "adapter_dir": str(merged_adapter_dir),
                        "sparsified_summary": sparsified_summary,
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
                    print(
                        f"Wrote DARE+TIES adapter for p={drop_probability} trim={trim_percentage}% "
                        f"lambda={merge_weight} to {merged_adapter_dir}"
                    )

    print(f"Saved DARE summary to {results_path}")


if __name__ == "__main__":
    main()
