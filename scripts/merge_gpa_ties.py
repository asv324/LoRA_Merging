"""GPA + TIES factor-space merging for LoRA adapters.

Pipeline:
1. Run GPA alignment on each module's `lora_A` factors.
2. Counter-rotate `lora_B` factors to preserve each adapter's effective delta.
3. Optionally fit directional GPA on unit-Frobenius `lora_A` factors.
4. Apply TIES only to the aligned `lora_A` factors, optionally with pre-TIES
   norm normalisation.
5. Average the aligned `lora_B` factors, optionally with inverse-norm weights.
6. Scale the merged adapter by lambda.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Literal, Sequence

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gpa_align_adapters import align_adapter_bundles
from scripts.merge_task_arithmetic import (
    TASKS,
    build_merged_adapter_config,
    copy_classifier_heads,
    copy_tokenizer_assets,
    evaluate_adapter_on_tasks,
    format_lambda,
    load_adapter_bundle,
    to_jsonable,
    validate_compatible_adapters,
)
from scripts.merge_ties import parse_trim_percentages, ties_merge_tensors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--trim-percentages",
        "--trim_percentages",
        dest="trim_percentages",
        default="10,20,30",
        help="Comma-separated trim percentages for TIES on aligned A factors.",
    )
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values applied after GPA+TIES.",
    )
    parser.add_argument(
        "--majority-sign-method",
        "--majority_sign_method",
        dest="majority_sign_method",
        choices=["total", "frequency"],
        default="total",
    )
    parser.add_argument("--max-iter", "--max_iter", dest="max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument(
        "--normalise-a-factors",
        "--normalise_a_factors",
        dest="normalise_a_factors",
        action="store_true",
        help="Run directional GPA by fitting rotations on unit-Frobenius A factors.",
    )
    parser.add_argument(
        "--scale-aware-ties",
        "--scale_aware_ties",
        dest="scale_aware_ties",
        action="store_true",
        help="Normalise aligned A factors before TIES and rescale the merge back to the average norm.",
    )
    parser.add_argument(
        "--b-weight-alpha",
        "--b_weight_alpha",
        dest="b_weight_alpha",
        type=float,
        default=0.0,
        help="Inverse-norm weighting exponent for aligned B averaging. 0.0 recovers the baseline mean.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "merged_adapters" / "gpa_ties",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "gpa_ties_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true")
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


def rank_pattern_from_reference(adapter_bundles: Sequence[Dict[str, object]]) -> Dict[str, int]:
    bundle = adapter_bundles[0]
    rank_pattern = bundle["config"].get("rank_pattern") or {}
    return {
        module_name: int(rank_pattern.get(module_name, bundle["config"]["r"]))
        for module_name in bundle["modules"]
    }


def _frobenius_norm(tensor: torch.Tensor) -> float:
    return float(tensor.float().norm(p="fro").item())


def ties_merge_scale_aware(
    aligned_a_factors: Sequence[torch.Tensor],
    *,
    density: float,
    majority_sign_method: Literal["total", "frequency"],
) -> tuple[torch.Tensor, List[float], float]:
    norms = [_frobenius_norm(matrix) for matrix in aligned_a_factors]
    if any(norm <= 0.0 for norm in norms):
        raise ValueError("scale-aware TIES requires all aligned A factors to have non-zero Frobenius norm")

    target_norm = sum(norms) / len(norms)
    rescaled = [
        matrix.float() / norm
        for matrix, norm in zip(aligned_a_factors, norms)
    ]
    weights = torch.ones(len(rescaled), dtype=torch.float32, device=rescaled[0].device)
    merged_unit = ties_merge_tensors(
        rescaled,
        weights=weights,
        density=density,
        majority_sign_method=majority_sign_method,
    ).float()
    merged = merged_unit * target_norm
    return merged.to(dtype=aligned_a_factors[0].dtype), norms, target_norm


def weighted_B_average(
    aligned_b_factors: Sequence[torch.Tensor],
    *,
    alpha: float,
) -> tuple[torch.Tensor, List[float], List[float]]:
    if alpha < 0.0:
        raise ValueError("b_weight_alpha must be non-negative")

    norms = [_frobenius_norm(matrix) for matrix in aligned_b_factors]
    if any(norm <= 0.0 for norm in norms):
        raise ValueError("inverse-norm B averaging requires all aligned B factors to have non-zero Frobenius norm")

    raw_weights = [(norm + 1e-8) ** (-alpha) for norm in norms]
    weight_total = sum(raw_weights)
    weights = [weight / weight_total for weight in raw_weights]
    merged = torch.zeros_like(aligned_b_factors[0], dtype=torch.float32)
    for weight, matrix in zip(weights, aligned_b_factors):
        merged = merged + (weight * matrix.float())
    return merged.to(dtype=aligned_b_factors[0].dtype), norms, weights


def build_variant_label(*, normalise_a_factors: bool, scale_aware_ties: bool, b_weight_alpha: float) -> str:
    alignment_label = "dGPA" if normalise_a_factors else "GPA"
    ties_label = "saTIES" if scale_aware_ties else "TIES"
    if b_weight_alpha > 0.0:
        return f"{alignment_label}+{ties_label}+wB({b_weight_alpha:g})"
    return f"{alignment_label}+{ties_label}"


def merge_gpa_ties_state_dicts(
    aligned_bundles: Sequence[Dict[str, object]],
    *,
    density: float,
    majority_sign_method: Literal["total", "frequency"],
    merge_weight: float,
    scale_aware_ties: bool = False,
    b_weight_alpha: float = 0.0,
) -> tuple[OrderedDict[str, torch.Tensor], List[Dict[str, object]]]:
    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    module_summaries: List[Dict[str, object]] = []

    modules = validate_compatible_adapters(aligned_bundles)
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        aligned_a_factors = [bundle["state_dict"][a_key] for bundle in aligned_bundles]
        aligned_b_factors = [bundle["state_dict"][b_key] for bundle in aligned_bundles]
        weights = torch.ones(len(aligned_a_factors), dtype=torch.float32, device=aligned_a_factors[0].device)

        if scale_aware_ties:
            merged_a, aligned_a_norms, merged_a_target_norm = ties_merge_scale_aware(
                aligned_a_factors,
                density=density,
                majority_sign_method=majority_sign_method,
            )
        else:
            merged_a = ties_merge_tensors(
                aligned_a_factors,
                weights=weights,
                density=density,
                majority_sign_method=majority_sign_method,
            )
            aligned_a_norms = [_frobenius_norm(matrix) for matrix in aligned_a_factors]
            merged_a_target_norm = None

        merged_b, aligned_b_norms, aligned_b_weights = weighted_B_average(
            aligned_b_factors,
            alpha=b_weight_alpha,
        )
        merged_a = (merged_a.float() * merge_weight).to(dtype=aligned_a_factors[0].dtype).contiguous()
        merged_b = merged_b.to(dtype=aligned_b_factors[0].dtype).contiguous()

        merged_state_dict[a_key] = merged_a
        merged_state_dict[b_key] = merged_b
        module_summaries.append(
            {
                "module_name": module_name,
                "merged_delta_frobenius_norm": float((merged_b.float() @ merged_a.float()).norm(p="fro").item()),
                "nonzero_fraction_merged_A": float((merged_a != 0).float().mean().item()),
                "avg_frobenius_norm_aligned_B": float(
                    sum(matrix.float().norm(p="fro").item() for matrix in aligned_b_factors) / len(aligned_b_factors)
                ),
                "scale_aware_ties": scale_aware_ties,
                "b_weight_alpha": b_weight_alpha,
                "aligned_A_frobenius_norms": aligned_a_norms,
                "scale_aware_ties_target_norm": merged_a_target_norm,
                "aligned_B_frobenius_norms": aligned_b_norms,
                "aligned_B_weights": aligned_b_weights,
            }
        )

    return merged_state_dict, module_summaries


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


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.b_weight_alpha < 0.0:
        raise ValueError("b_weight_alpha must be non-negative")

    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    tasks = list(args.tasks)
    trim_percentages = parse_trim_percentages(args.trim_percentages)
    lambda_values = parse_lambda_values(args.lambdas)
    variant_label = build_variant_label(
        normalise_a_factors=args.normalise_a_factors,
        scale_aware_ties=args.scale_aware_ties,
        b_weight_alpha=args.b_weight_alpha,
    )

    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    aligned_bundles, alignment_diagnostics = align_adapter_bundles(
        adapter_bundles,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        normalise=args.normalise_a_factors,
    )

    merged_rank_pattern = rank_pattern_from_reference(adapter_bundles)
    merged_config = build_merged_adapter_config(adapter_bundles[0]["config"], merged_rank_pattern)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "gpa_ties_factor_space",
        "source_tasks": tasks,
        "trim_percentages": trim_percentages,
        "lambda_values": lambda_values,
        "majority_sign_method": args.majority_sign_method,
        "variant_label": variant_label,
        "scale_aware_ties": args.scale_aware_ties,
        "b_weight_alpha": args.b_weight_alpha,
        "gpa": {
            "max_iter": args.max_iter,
            "tol": args.tol,
            "init": args.init,
            "normalise_a_factors": args.normalise_a_factors,
            "module_diagnostics": alignment_diagnostics,
        },
        "runs": [],
    }

    for trim_percentage in trim_percentages:
        density = 1.0 - (trim_percentage / 100.0)
        for merge_weight in lambda_values:
            merged_state_dict, module_summaries = merge_gpa_ties_state_dicts(
                aligned_bundles,
                density=density,
                majority_sign_method=args.majority_sign_method,
                merge_weight=merge_weight,
                scale_aware_ties=args.scale_aware_ties,
                b_weight_alpha=args.b_weight_alpha,
            )
            merged_adapter_dir = output_dir / f"trim_{trim_percentage}" / f"lambda_{format_lambda(merge_weight)}"
            metadata = {
                "method": "gpa_ties_factor_space",
                "variant_label": variant_label,
                "source_tasks": tasks,
                "trim_percentage": trim_percentage,
                "density": density,
                "lambda": merge_weight,
                "majority_sign_method": args.majority_sign_method,
                "scale_aware_ties": args.scale_aware_ties,
                "b_weight_alpha": args.b_weight_alpha,
                "gpa": {
                    "max_iter": args.max_iter,
                    "tol": args.tol,
                    "init": args.init,
                    "normalise_a_factors": args.normalise_a_factors,
                },
                "module_summaries": module_summaries,
            }
            write_merged_adapter(
                output_dir=merged_adapter_dir,
                merged_state_dict=merged_state_dict,
                merged_config=merged_config,
                reference_adapter_dir=Path(adapter_bundles[0]["adapter_dir"]),
                metadata=metadata,
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
            print(f"Wrote GPA+TIES adapter for trim={trim_percentage}% lambda={merge_weight} to {merged_adapter_dir}")

    print(f"Saved GPA+TIES summary to {results_path}")


if __name__ == "__main__":
    main()
