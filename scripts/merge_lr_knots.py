"""LR-KnOTS + TIES baseline for LoRA adapters.

This implementation adapts the KnOTS idea to the Week 2 factor-space pipeline:

1. Stack all `lora_A` factors for a module into `A_concat` with shape `(N*r, d)`.
2. Compute `A_concat = U Sigma V^T`.
3. Split `U Sigma` into per-adapter latent blocks of shape `(r, k)`.
4. Apply TIES to those latent blocks.
5. Reconstruct `A_merged = latent_merged @ V^T`.
6. Average the raw `lora_B` factors.

This is a LoRA-friendly factor-space approximation of KnOTS rather than the
full delta-space method from the paper. It keeps the result loadable as a LoRA
adapter and makes the comparison to GPA+TIES straightforward in this project.
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
    parse_lambda_values,
    to_jsonable,
    validate_compatible_adapters,
)
from scripts.merge_ties import parse_trim_percentages, rank_pattern_from_reference, ties_merge_tensors

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
        "--trim-percentages",
        "--trim_percentages",
        dest="trim_percentages",
        default="10,20,30",
        help="Comma-separated trim percentages for the TIES step in latent space.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values applied after reconstruction.",
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
        default=PROJECT_ROOT / "merged_adapters" / "lr_knots",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "lr_knots_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    return parser


def copy_tokenizer_assets(reference_adapter_dir: Path, output_dir: Path) -> None:
    for asset_name in TOKENIZER_ASSETS:
        source_path = reference_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)


def decompose_stacked_a_factors(a_matrices: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    if not a_matrices:
        raise ValueError("at least one A matrix is required")

    rank = int(a_matrices[0].shape[0])
    a_concat = torch.cat([matrix.float() for matrix in a_matrices], dim=0)
    u, singular_values, vh = torch.linalg.svd(a_concat, full_matrices=False)
    latent_concat = u * singular_values.unsqueeze(0)
    latent_blocks = list(torch.split(latent_concat, rank, dim=0))
    return latent_blocks, vh, singular_values


def merge_lr_knots_state_dicts(
    adapter_bundles: Sequence[Dict[str, object]],
    *,
    density: float,
    majority_sign_method: str,
    merge_weight: float,
) -> tuple[OrderedDict[str, torch.Tensor], List[Dict[str, object]]]:
    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    module_summaries: List[Dict[str, object]] = []

    modules = validate_compatible_adapters(adapter_bundles)
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        a_matrices = [bundle["state_dict"][a_key] for bundle in adapter_bundles]
        b_matrices = [bundle["state_dict"][b_key] for bundle in adapter_bundles]
        latent_blocks, shared_basis, singular_values = decompose_stacked_a_factors(a_matrices)

        weights = torch.ones(len(latent_blocks), dtype=torch.float32, device=latent_blocks[0].device)
        merged_latent = ties_merge_tensors(
            latent_blocks,
            weights=weights,
            density=density,
            majority_sign_method=majority_sign_method,
        )
        merged_a = (merged_latent.float() @ shared_basis.float()) * merge_weight
        merged_b = torch.mean(torch.stack([matrix.float() for matrix in b_matrices], dim=0), dim=0)

        target_a_dtype = a_matrices[0].dtype
        target_b_dtype = b_matrices[0].dtype
        merged_a = merged_a.to(dtype=target_a_dtype).contiguous()
        merged_b = merged_b.to(dtype=target_b_dtype).contiguous()

        merged_state_dict[a_key] = merged_a
        merged_state_dict[b_key] = merged_b

        reconstruction_errors = []
        for block, matrix in zip(latent_blocks, a_matrices):
            reconstructed = block.float() @ shared_basis.float()
            reconstruction_errors.append(float((reconstructed - matrix.float()).norm(p="fro").item()))

        merged_delta_norm = float((merged_b.float() @ merged_a.float()).norm(p="fro").item())
        module_summaries.append(
            {
                "module_name": module_name,
                "shared_rank": int(shared_basis.shape[0]),
                "mean_reconstruction_error": float(sum(reconstruction_errors) / len(reconstruction_errors)),
                "max_reconstruction_error": max(reconstruction_errors),
                "leading_singular_values": [float(value.item()) for value in singular_values[: min(5, singular_values.numel())]],
                "merged_delta_frobenius_norm": merged_delta_norm,
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
        "# LR-KnOTS Merged Adapter\n\n"
        f"Merged source adapters: {', '.join(tasks)}\n\n"
        f"Trim percentage: {trim_percentage}\n\n"
        f"Density kept after trimming: {density}\n\n"
        f"Lambda: {merge_weight}\n\n"
        f"Majority sign method: {majority_sign_method}\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    metadata = {
        "method": "lr_knots_factor_space_ties",
        "source_tasks": list(tasks),
        "trim_percentage": trim_percentage,
        "density": density,
        "lambda": merge_weight,
        "majority_sign_method": majority_sign_method,
        "factor_space_limitation": "This baseline applies KnOTS only to stacked lora_A factors and averages lora_B factors, which is a LoRA-friendly approximation of the full delta-space KnOTS method.",
        "module_summaries": list(module_summaries),
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    tasks = list(args.tasks)
    trim_percentages = parse_trim_percentages(args.trim_percentages)
    lambda_values = parse_lambda_values(args.lambdas)

    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    validate_compatible_adapters(adapter_bundles)

    merged_rank_pattern = rank_pattern_from_reference(adapter_bundles)
    merged_config = build_merged_adapter_config(adapter_bundles[0]["config"], merged_rank_pattern)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "lr_knots_factor_space_ties",
        "source_tasks": tasks,
        "trim_percentages": trim_percentages,
        "lambda_values": lambda_values,
        "majority_sign_method": args.majority_sign_method,
        "runs": [],
    }

    for trim_percentage in trim_percentages:
        density = 1.0 - (trim_percentage / 100.0)
        for merge_weight in lambda_values:
            merged_state_dict, module_summaries = merge_lr_knots_state_dicts(
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
            print(f"Wrote LR-KnOTS adapter for trim={trim_percentage}% lambda={merge_weight} to {merged_adapter_dir}")

    print(f"Saved LR-KnOTS summary to {results_path}")


if __name__ == "__main__":
    main()
