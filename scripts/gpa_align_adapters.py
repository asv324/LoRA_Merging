"""Apply GPA alignment to real LoRA adapters.

For each LoRA module across a set of adapters:
1. Load all `lora_A` factors.
2. Run GPA to estimate orthogonal rotations `Q_i`.
3. Construct aligned factors `A_tilde_i = Q_i @ A_i` and `B_tilde_i = B_i @ Q_i^T`.
4. Save functionally equivalent aligned adapters and layer diagnostics.
"""

from __future__ import annotations

import argparse
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

from scripts.gpa import gpa_align
from scripts.merge_task_arithmetic import TASKS, load_adapter_bundle, to_jsonable, validate_compatible_adapters

TOKENIZER_ASSETS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "README.md",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters-dir", "--adapters_dir", dest="adapters_dir", default=PROJECT_ROOT / "adapters")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=PROJECT_ROOT / "aligned_adapters" / "gpa",
    )
    parser.add_argument(
        "--results-path",
        "--results_path",
        dest="results_path",
        default=PROJECT_ROOT / "results" / "gpa_alignment_diagnostics.json",
    )
    parser.add_argument("--max-iter", "--max_iter", dest="max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--init", choices=["first", "mean"], default="first")
    parser.add_argument(
        "--normalise-a-factors",
        "--normalise_a_factors",
        dest="normalise_a_factors",
        action="store_true",
        help="Fit GPA rotations on unit-Frobenius A factors before applying them to the original matrices.",
    )
    return parser


def copy_supporting_files(reference_adapter_dir: Path, output_dir: Path) -> None:
    for asset_name in TOKENIZER_ASSETS:
        source_path = reference_adapter_dir / asset_name
        if source_path.exists():
            shutil.copy2(source_path, output_dir / asset_name)


def align_module_factors(
    a_matrices: Sequence[torch.Tensor],
    b_matrices: Sequence[torch.Tensor],
    *,
    max_iter: int,
    tol: float,
    init: str,
    normalise: bool = False,
) -> tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, object]]:
    if len(a_matrices) != len(b_matrices):
        raise ValueError("A and B matrix lists must have the same length")

    a_arrays = [matrix.detach().cpu().float().numpy() for matrix in a_matrices]
    rotations_np, _, residuals = gpa_align(
        a_arrays,
        max_iter=max_iter,
        tol=tol,
        init=init,
        normalise=normalise,
    )

    aligned_a_matrices: List[torch.Tensor] = []
    aligned_b_matrices: List[torch.Tensor] = []
    per_adapter_diagnostics = []

    for index, (a_matrix, b_matrix, rotation_np) in enumerate(zip(a_matrices, b_matrices, rotations_np)):
        rotation = torch.from_numpy(rotation_np).to(device=a_matrix.device, dtype=torch.float32)
        aligned_a = rotation @ a_matrix.float()
        aligned_b = b_matrix.float() @ rotation.T

        before_norm = float(a_matrix.float().norm(p="fro").item())
        after_norm = float(aligned_a.norm(p="fro").item())
        invariance_error = float((aligned_b @ aligned_a - b_matrix.float() @ a_matrix.float()).norm(p="fro").item())
        orthogonality_error = float((rotation.T @ rotation - torch.eye(rotation.shape[0], dtype=torch.float32, device=rotation.device)).norm(p="fro").item())

        aligned_a_matrices.append(aligned_a.to(dtype=a_matrix.dtype).contiguous())
        aligned_b_matrices.append(aligned_b.to(dtype=b_matrix.dtype).contiguous())
        per_adapter_diagnostics.append(
            {
                "adapter_index": index,
                "frobenius_norm_before": before_norm,
                "frobenius_norm_after": after_norm,
                "norm_difference": abs(after_norm - before_norm),
                "functional_invariance_error": invariance_error,
                "orthogonality_error": orthogonality_error,
            }
        )

    aligned_a_float = [matrix.float() for matrix in aligned_a_matrices]
    consensus_tensor = torch.mean(torch.stack(aligned_a_float, dim=0), dim=0)
    alignment_residual = float(
        sum((aligned_matrix - consensus_tensor).norm(p="fro").item() ** 2 for aligned_matrix in aligned_a_float)
    )

    diagnostics = {
        "iterations": len(residuals),
        "normalised_alignment": normalise,
        "final_alignment_residual": alignment_residual,
        "optimisation_residual": float(residuals[-1]) if residuals else 0.0,
        "consensus_frobenius_norm": float(consensus_tensor.norm(p="fro").item()),
        "per_adapter": per_adapter_diagnostics,
    }
    return aligned_a_matrices, aligned_b_matrices, diagnostics


def align_adapter_bundles(
    adapter_bundles: Sequence[Dict[str, object]],
    *,
    max_iter: int,
    tol: float,
    init: str,
    normalise: bool = False,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    modules = validate_compatible_adapters(adapter_bundles)

    aligned_bundles: List[Dict[str, object]] = []
    for bundle in adapter_bundles:
        aligned_bundles.append(
            {
                "task": bundle["task"],
                "adapter_dir": bundle["adapter_dir"],
                "config": json.loads(json.dumps(bundle["config"])),
                "state_dict": OrderedDict((key, value.clone()) for key, value in bundle["state_dict"].items()),
                "modules": list(bundle["modules"]),
            }
        )

    module_diagnostics = []
    for module_name in modules:
        a_key = f"{module_name}.lora_A.weight"
        b_key = f"{module_name}.lora_B.weight"
        a_matrices = [bundle["state_dict"][a_key] for bundle in adapter_bundles]
        b_matrices = [bundle["state_dict"][b_key] for bundle in adapter_bundles]

        aligned_a_matrices, aligned_b_matrices, diagnostics = align_module_factors(
            a_matrices,
            b_matrices,
            max_iter=max_iter,
            tol=tol,
            init=init,
            normalise=normalise,
        )

        for bundle, aligned_a, aligned_b in zip(aligned_bundles, aligned_a_matrices, aligned_b_matrices):
            bundle["state_dict"][a_key] = aligned_a
            bundle["state_dict"][b_key] = aligned_b

        module_diagnostics.append(
            {
                "module_name": module_name,
                **diagnostics,
            }
        )

    return aligned_bundles, module_diagnostics


def write_aligned_adapters(
    aligned_bundles: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    tasks: Sequence[str],
    max_iter: int,
    tol: float,
    init: str,
    normalise: bool,
) -> List[Dict[str, object]]:
    written = []
    for bundle in aligned_bundles:
        task_output_dir = output_dir / bundle["task"]
        task_output_dir.mkdir(parents=True, exist_ok=True)
        save_file(dict(bundle["state_dict"]), str(task_output_dir / "adapter_model.safetensors"))
        (task_output_dir / "adapter_config.json").write_text(json.dumps(bundle["config"], indent=2), encoding="utf-8")
        copy_supporting_files(Path(bundle["adapter_dir"]), task_output_dir)
        (task_output_dir / "alignment_metadata.json").write_text(
            json.dumps(
                {
                    "method": "gpa_alignment",
                    "task": bundle["task"],
                    "source_tasks": list(tasks),
                    "max_iter": max_iter,
                    "tol": tol,
                    "init": init,
                    "normalise_a_factors": normalise,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append({"task": bundle["task"], "adapter_dir": str(task_output_dir)})
    return written


def main() -> None:
    args = build_arg_parser().parse_args()
    adapters_dir = Path(args.adapters_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)
    tasks = list(args.tasks)

    adapter_bundles = [load_adapter_bundle(adapters_dir / task) for task in tasks]
    aligned_bundles, module_diagnostics = align_adapter_bundles(
        adapter_bundles,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        normalise=args.normalise_a_factors,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    written_adapters = write_aligned_adapters(
        aligned_bundles,
        output_dir=output_dir,
        tasks=tasks,
        max_iter=args.max_iter,
        tol=args.tol,
        init=args.init,
        normalise=args.normalise_a_factors,
    )

    results_payload = {
        "method": "gpa_alignment",
        "source_tasks": tasks,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "init": args.init,
        "normalise_a_factors": args.normalise_a_factors,
        "aligned_adapters": written_adapters,
        "module_diagnostics": module_diagnostics,
    }
    results_path.write_text(json.dumps(to_jsonable(results_payload), indent=2), encoding="utf-8")
    print(f"Saved aligned adapters to {output_dir}")
    print(f"Saved GPA diagnostics to {results_path}")


if __name__ == "__main__":
    main()
