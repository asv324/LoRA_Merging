"""Task Arithmetic in aligned factor space (Step 4.2 / Experiment 12).

Pipeline:

1. Run GPA (or directional dGPA) alignment on each module's ``lora_A`` factors
   and counter-rotate the paired ``lora_B`` factors to preserve each adapter's
   effective delta.
2. Merge the aligned adapters by summing the reconstructed deltas
   ``B_tilde_i @ A_tilde_i`` - i.e. Task Arithmetic applied to aligned factors
   instead of raw ones.
3. Optionally reweight each adapter's contribution by the inverse Frobenius
   norm of its aligned ``B`` factor raised to a user-specified exponent; the
   per-adapter weights are renormalised to average to 1 so their overall scale
   matches the unweighted (alpha=0) sum.

This supports Experiment 12 in the dissertation methodology (§3.3.4), which
asks for two rows in the main results table:

- ``GPA-aligned TA``: plain GPA alignment + unweighted task arithmetic
  (alpha = 0.0, normalise_a_factors = False). Because GPA preserves each
  adapter's effective delta, this should be numerically identical to the
  unaligned Task Arithmetic baseline (up to floating-point noise). Running
  it explicitly is the sanity-check the methodology asks for.
- ``enhanced-GPA-aligned TA``: directional dGPA alignment + inverse-norm
  ``B`` weighting (alpha > 0, normalise_a_factors = True). This breaks the
  delta-preserving invariance and is the substantive Experiment 12 row.

The merged adapter is still a valid LoRA adapter because the merged factors
are stored by concatenating the per-adapter aligned blocks:

    B_merge = [B_tilde_1, ..., B_tilde_N]
    A_merge = [lambda * scale_1 * w_1 * A_tilde_1;
               ...
               lambda * scale_N * w_N * A_tilde_N]

so ``B_merge @ A_merge = lambda * sum_i scale_i * w_i * B_tilde_i @ A_tilde_i``
reproduces the intended weighted sum of aligned deltas exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

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
    resolve_module_scale,
    to_jsonable,
    validate_compatible_adapters,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--lambdas",
        default="1.0",
        help="Comma-separated lambda values applied after aligned-space TA. Defaults to 1.0 (the best lambda for raw TA in Experiment 6).",
    )
    parser.add_argument("--max-iter", "--max_iter", dest="max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument(
        "--normalise-a-factors",
        "--normalise_a_factors",
        dest="normalise_a_factors",
        action="store_true",
        help="Fit GPA rotations on unit-Frobenius A factors (directional GPA / dGPA).",
    )
    parser.add_argument(
        "--b-weight-alpha",
        "--b_weight_alpha",
        dest="b_weight_alpha",
        type=float,
        default=0.0,
        help=(
            "Inverse-norm weighting exponent for per-adapter contributions. "
            "Weights w_i are set to N * (|B_tilde_i|_F + eps)^-alpha / sum_j (|B_tilde_j|_F + eps)^-alpha, "
            "so the weights average to 1 and alpha=0 recovers plain Task Arithmetic on aligned deltas."
        ),
    )
    parser.add_argument(
        "--variant-label",
        "--variant_label",
        dest="variant_label",
        default=None,
        help="Optional human-readable label stored in the run metadata; defaults to an auto-generated label.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "merged_adapters" / "ta_aligned",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "ablation_ta_aligned" / "ta_aligned_results.json",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=32)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", dest="max_eval_samples", type=int, default=None)
    return parser


def parse_lambda_values(raw_value: str) -> List[float]:
    values: List[float] = []
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


def rank_pattern_from_concat(
    adapter_bundles: Sequence[Dict[str, object]],
    modules: Sequence[str],
) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for module_name in modules:
        total_rank = 0
        for bundle in adapter_bundles:
            a_key = f"{module_name}.lora_A.weight"
            total_rank += int(bundle["state_dict"][a_key].shape[0])
        merged[module_name] = total_rank
    return merged


def _frobenius_norm(tensor: torch.Tensor) -> float:
    return float(tensor.float().norm(p="fro").item())


def compute_per_adapter_weights(
    aligned_b_factors: Sequence[torch.Tensor],
    *,
    alpha: float,
) -> tuple[List[float], List[float]]:
    """Return (|B_tilde_i|_F, w_i) where the w_i average to 1.

    alpha = 0 yields w_i = 1 for all i (plain TA on aligned deltas). alpha > 0
    redistributes mass towards adapters with smaller aligned-B norms while
    preserving the overall scale of the sum.
    """
    if alpha < 0.0:
        raise ValueError("b_weight_alpha must be non-negative")

    norms = [_frobenius_norm(matrix) for matrix in aligned_b_factors]
    if any(norm <= 0.0 for norm in norms):
        raise ValueError("inverse-norm B weighting requires all aligned B factors to have non-zero Frobenius norm")

    if alpha == 0.0:
        return norms, [1.0 for _ in norms]

    n = len(norms)
    raw_weights = [(norm + 1e-8) ** (-alpha) for norm in norms]
    total = sum(raw_weights)
    weights = [n * raw / total for raw in raw_weights]
    return norms, weights


def build_variant_label(*, normalise_a_factors: bool, b_weight_alpha: float) -> str:
    alignment_label = "dGPA" if normalise_a_factors else "GPA"
    if b_weight_alpha > 0.0:
        return f"{alignment_label}-aligned TA + wB({b_weight_alpha:g})"
    return f"{alignment_label}-aligned TA"


def merge_ta_aligned_state_dicts(
    aligned_bundles: Sequence[Dict[str, object]],
    *,
    merge_weight: float,
    b_weight_alpha: float = 0.0,
) -> tuple[OrderedDict[str, torch.Tensor], Dict[str, int], List[Dict[str, object]]]:
    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    merged_rank_pattern: Dict[str, int] = {}
    module_summaries: List[Dict[str, object]] = []

    modules = validate_compatible_adapters(aligned_bundles)
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"

        aligned_a_factors = [bundle["state_dict"][a_key] for bundle in aligned_bundles]
        aligned_b_factors = [bundle["state_dict"][b_key] for bundle in aligned_bundles]

        aligned_b_norms, per_adapter_weights = compute_per_adapter_weights(
            aligned_b_factors,
            alpha=b_weight_alpha,
        )

        reference_a_dtype = aligned_a_factors[0].dtype
        reference_b_dtype = aligned_b_factors[0].dtype
        reference_device = aligned_a_factors[0].device

        weighted_a_blocks: List[torch.Tensor] = []
        b_blocks: List[torch.Tensor] = []
        source_ranks: List[int] = []
        per_adapter_scales: List[float] = []
        for bundle, aligned_a, aligned_b, weight in zip(
            aligned_bundles, aligned_a_factors, aligned_b_factors, per_adapter_weights
        ):
            module_scale = resolve_module_scale(bundle["config"], module_name)
            block_scale = merge_weight * module_scale * weight
            weighted_a_blocks.append(aligned_a.to(dtype=torch.float32) * block_scale)
            b_blocks.append(aligned_b.to(dtype=torch.float32))
            source_ranks.append(int(aligned_a.shape[0]))
            per_adapter_scales.append(float(module_scale))

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
                "per_adapter_scales": per_adapter_scales,
                "per_adapter_weights": per_adapter_weights,
                "aligned_B_frobenius_norms": aligned_b_norms,
                "b_weight_alpha": b_weight_alpha,
            }
        )

    return merged_state_dict, merged_rank_pattern, module_summaries


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
    lambda_values = parse_lambda_values(args.lambdas)
    variant_label = args.variant_label or build_variant_label(
        normalise_a_factors=args.normalise_a_factors,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "method": "ta_aligned_factor_space",
        "variant_label": variant_label,
        "source_tasks": tasks,
        "lambda_values": lambda_values,
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

    for merge_weight in lambda_values:
        merged_state_dict, merged_rank_pattern, module_summaries = merge_ta_aligned_state_dicts(
            aligned_bundles,
            merge_weight=merge_weight,
            b_weight_alpha=args.b_weight_alpha,
        )
        merged_config = build_merged_adapter_config(adapter_bundles[0]["config"], merged_rank_pattern)
        merged_adapter_dir = output_dir / f"lambda_{format_lambda(merge_weight)}"
        metadata = {
            "method": "ta_aligned_factor_space",
            "variant_label": variant_label,
            "source_tasks": tasks,
            "lambda": merge_weight,
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
        print(f"Wrote aligned-TA adapter for lambda={merge_weight} to {merged_adapter_dir}")

    print(f"Saved aligned-TA summary to {results_path}")


if __name__ == "__main__":
    main()
