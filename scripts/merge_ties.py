"""TIES-Merging baseline for raw, unaligned LoRA weight matrices.

This implementation applies TIES in factor space on the raw LoRA factors:

1. Trim the smallest-magnitude parameters in each task tensor.
2. Elect the majority sign from the trimmed, unweighted tensors.
3. Merge only values that agree with the elected sign.

The merge is applied independently to each raw `lora_A` and `lora_B` tensor.
Per-adapter LoRA scaling is folded into the factor weights as:

    factor_weight_i = sqrt(alpha_i / r_i)

These weights are applied before sign election so the elected sign reflects the
scaled task contribution when adapters have different `alpha / r` values.

After TIES, the merged adapter is scaled by lambda by multiplying the merged
`lora_A` tensors. This scales the effective LoRA delta by lambda while keeping
the saved adapter config at unit runtime scaling (`lora_alpha = r`).

Important limitation: this is still factor-space TIES, not full-delta TIES on
`DeltaW = (alpha / r) * B @ A`. That makes it loadable as a LoRA adapter, but
it does not remove the factor-separability confound discussed in the KnOTS
literature.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.merge_task_arithmetic import (
    TASKS,
    build_merged_adapter_config,
    copy_classifier_heads,
    format_lambda,
    load_adapter_bundle,
    to_jsonable,
    validate_compatible_adapters,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--trim-percentages",
        "--trim_percentages",
        dest="trim_percentages",
        default="10,20,30",
        help="Comma-separated trim percentages k, where the bottom k%% by magnitude are zeroed.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values applied after TIES.",
    )
    parser.add_argument(
        "--majority-sign-method",
        "--majority_sign_method",
        dest="majority_sign_method",
        choices=["total", "frequency"],
        default="total",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "merged_adapters" / "ties",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "ties_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    return parser


def parse_float_list(raw_value: str, *, non_negative: bool = True) -> List[float]:
    values = []
    for piece in raw_value.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        value = float(stripped)
        if non_negative and value < 0:
            raise ValueError("values must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("at least one value is required")
    return values


def parse_trim_percentages(raw_value: str) -> List[int]:
    values = []
    for value in parse_float_list(raw_value):
        if value < 0 or value > 100:
            raise ValueError("trim percentages must lie in [0, 100]")
        if not float(value).is_integer():
            raise ValueError("trim percentages must be integers")
        values.append(int(value))
    return values


def reshape_weight_task_tensors(task_tensors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    new_shape = weights.shape + (1,) * (task_tensors.dim() - weights.dim())
    return weights.view(new_shape)


def magnitude_prune(tensor: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return tensor.clone()
    if density < 0.0:
        raise ValueError(f"density must be non-negative, got {density}")

    flat = tensor.reshape(-1)
    keep_count = int(density * flat.numel())
    if keep_count <= 0:
        return torch.zeros_like(tensor)

    mask = torch.zeros_like(flat)
    topk = torch.topk(flat.abs(), k=keep_count, largest=True)
    mask[topk.indices] = 1
    return tensor * mask.reshape(tensor.shape)


def calculate_majority_sign_mask(
    task_tensors: torch.Tensor,
    method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    signs = task_tensors.sign()
    if method == "total":
        sign_magnitude = task_tensors.sum(dim=0)
    elif method == "frequency":
        sign_magnitude = signs.sum(dim=0)
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unsupported majority sign method: {method}")
    majority_sign = torch.where(sign_magnitude >= 0, 1, -1)
    return signs == majority_sign


def disjoint_merge(weighted_task_tensors: torch.Tensor, majority_sign_mask: torch.Tensor) -> torch.Tensor:
    merged = (weighted_task_tensors * majority_sign_mask).sum(dim=0)
    preserved = majority_sign_mask.sum(dim=0)
    return merged / torch.clamp(preserved, min=1.0)


def ties_merge_tensors(
    task_tensors: Sequence[torch.Tensor],
    weights: torch.Tensor,
    density: float,
    majority_sign_method: Literal["total", "frequency"] = "total",
) -> torch.Tensor:
    pruned = [magnitude_prune(tensor.float(), density=density) for tensor in task_tensors]
    stacked = torch.stack(pruned, dim=0)
    weighted = stacked * reshape_weight_task_tensors(stacked, weights.to(stacked.device, dtype=stacked.dtype))
    majority_sign_mask = calculate_majority_sign_mask(weighted, method=majority_sign_method)
    merged = disjoint_merge(weighted, majority_sign_mask)
    return merged.to(dtype=task_tensors[0].dtype)


def resolve_module_rank(config: Dict[str, object], module_name: str) -> int:
    rank_pattern = config.get("rank_pattern") or {}
    return int(rank_pattern.get(module_name, config["r"]))


def resolve_module_alpha(config: Dict[str, object], module_name: str) -> float:
    alpha_pattern = config.get("alpha_pattern") or {}
    return float(alpha_pattern.get(module_name, config["lora_alpha"]))


def resolve_factor_weight(config: Dict[str, object], module_name: str) -> float:
    rank = resolve_module_rank(config, module_name)
    alpha = resolve_module_alpha(config, module_name)
    if rank <= 0:
        raise ValueError(f"rank must be positive for module {module_name}")
    return math.sqrt(alpha / rank)


def rank_pattern_from_reference(adapter_bundles: Sequence[Dict[str, object]]) -> Dict[str, int]:
    bundle = adapter_bundles[0]
    return {
        module_name: resolve_module_rank(bundle["config"], module_name)
        for module_name in bundle["modules"]
    }


def copy_tokenizer_assets(reference_adapter_dir: Path, output_dir: Path) -> None:
    for asset_name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
    ):
        source_path = reference_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)


def merge_ties_state_dicts(
    adapter_bundles: Sequence[Dict[str, object]],
    *,
    density: float,
    majority_sign_method: Literal["total", "frequency"],
    merge_weight: float,
) -> tuple[OrderedDict[str, torch.Tensor], List[Dict[str, object]]]:
    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    module_summaries: List[Dict[str, object]] = []

    modules = validate_compatible_adapters(adapter_bundles)
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        lora_a_list = []
        lora_b_list = []
        factor_weights = []
        for bundle in adapter_bundles:
            lora_a_list.append(bundle["state_dict"][a_key])
            lora_b_list.append(bundle["state_dict"][b_key])
            factor_weights.append(resolve_factor_weight(bundle["config"], module_name))

        weights = torch.tensor(factor_weights, dtype=torch.float32, device=lora_a_list[0].device)
        merged_lora_a = ties_merge_tensors(
            lora_a_list,
            weights=weights,
            density=density,
            majority_sign_method=majority_sign_method,
        )
        merged_lora_b = ties_merge_tensors(
            lora_b_list,
            weights=weights,
            density=density,
            majority_sign_method=majority_sign_method,
        )
        merged_lora_a = (merged_lora_a.float() * merge_weight).to(dtype=lora_a_list[0].dtype)

        merged_state_dict[a_key] = merged_lora_a.contiguous()
        merged_state_dict[b_key] = merged_lora_b.contiguous()
        merged_delta_norm = float((merged_lora_b.float() @ merged_lora_a.float()).norm(p="fro").item())
        kept_a_fraction = float((merged_lora_a != 0).float().mean().item())
        kept_b_fraction = float((merged_lora_b != 0).float().mean().item())
        module_summaries.append(
            {
                "module_name": module_name,
                "factor_weights": factor_weights,
                "merged_delta_frobenius_norm": merged_delta_norm,
                "nonzero_fraction_lora_A": kept_a_fraction,
                "nonzero_fraction_lora_B": kept_b_fraction,
            }
        )

    return merged_state_dict, module_summaries


def write_merged_adapter(
    *,
    output_dir: Path,
    merged_state_dict: OrderedDict[str, torch.Tensor],
    merged_config: Dict[str, object],
    reference_adapter_dir: Path,
    tasks: Sequence[str],
    trim_percentage: int,
    density: float,
    merge_weight: float,
    majority_sign_method: str,
    module_summaries: Sequence[Dict[str, object]],
    adapter_bundles: Sequence[Dict[str, object]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(dict(merged_state_dict), str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(json.dumps(merged_config, indent=2), encoding="utf-8")
    copy_tokenizer_assets(reference_adapter_dir, output_dir)
    if adapter_bundles is not None:
        copy_classifier_heads(adapter_bundles, output_dir)

    readme = (
        "---\n"
        "library_name: peft\n"
        "---\n\n"
        "# TIES Merged Adapter\n\n"
        f"Merged source adapters: {', '.join(tasks)}\n\n"
        f"Trim percentage: {trim_percentage}\n\n"
        f"Density kept after trimming: {density}\n\n"
        f"Lambda: {merge_weight}\n\n"
        f"Majority sign method: {majority_sign_method}\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    metadata = {
        "method": "ties_factor_space_raw_lora_factors",
        "source_tasks": list(tasks),
        "trim_percentage": trim_percentage,
        "density": density,
        "lambda": merge_weight,
        "majority_sign_method": majority_sign_method,
        "why_factor_weights_use_sqrt_scaling": "Weighting both A and B by sqrt(alpha/r) preserves the effective LoRA delta scale when working in factor space.",
        "factor_space_limitation": "This baseline applies TIES independently to A and B factors rather than to full DeltaW matrices, so it remains a pragmatic factor-space approximation rather than exact delta-space TIES.",
        "module_summaries": list(module_summaries),
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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
    tasks = list(args.tasks)
    trim_percentages = parse_trim_percentages(args.trim_percentages)
    lambda_values = parse_float_list(args.lambdas)

    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    validate_compatible_adapters(adapter_bundles)

    merged_rank_pattern = rank_pattern_from_reference(adapter_bundles)
    merged_config = build_merged_adapter_config(adapter_bundles[0]["config"], merged_rank_pattern)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "ties_factor_space_raw_lora_factors",
        "source_tasks": tasks,
        "trim_percentages": trim_percentages,
        "lambda_values": lambda_values,
        "majority_sign_method": args.majority_sign_method,
        "runs": [],
    }

    for trim_percentage in trim_percentages:
        density = 1.0 - (trim_percentage / 100.0)
        for merge_weight in lambda_values:
            merged_state_dict, module_summaries = merge_ties_state_dicts(
                adapter_bundles,
                density=density,
                majority_sign_method=args.majority_sign_method,
                merge_weight=merge_weight,
            )
            merged_adapter_dir = output_dir / f"trim_{trim_percentage}" / f"lambda_{format_lambda(merge_weight)}"
            write_merged_adapter(
                output_dir=merged_adapter_dir,
                merged_state_dict=merged_state_dict,
                merged_config=merged_config,
                reference_adapter_dir=Path(adapter_bundles[0]["adapter_dir"]),
                tasks=tasks,
                trim_percentage=trim_percentage,
                density=density,
                merge_weight=merge_weight,
                majority_sign_method=args.majority_sign_method,
                module_summaries=module_summaries,
                adapter_bundles=adapter_bundles,
            )

            run_summary = {
                "trim_percentage": trim_percentage,
                "density": density,
                "lambda": merge_weight,
                "adapter_dir": str(merged_adapter_dir),
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
                f"Wrote TIES adapter for trim={trim_percentage}% lambda={merge_weight} "
                f"to {merged_adapter_dir}"
            )

    print(f"Saved TIES summary to {results_path}")


if __name__ == "__main__":
    main()
