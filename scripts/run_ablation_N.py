"""Step 4.1 / Experiment 11 driver: vary the number of merged adapters.

For each N in {2, 3, 4, 5} and each of four merging methods (GPA+TIES, the
enhanced ``dGPA+saTIES+wB(0.5)`` variant, raw TIES, and LR-KnOTS+TIES), this
script merges every selected task subset using the best per-method
hyperparameters from Week 3 (sourced from ``results/best_hparams.json``) and
evaluates the merged adapter on exactly the tasks that were merged.

The per-configuration JSONs produced by the existing merge scripts are kept
unchanged under ``results/ablation_N/N_{N}/{method}/{subset_id}.json``. The
N = 5 case is not rerun here: the restored-head Week 3 argmax rows already
live in ``results/best_hparams.json`` and are linked into the summary step
(``analyze_ablation_N.py``) as the full five-task merge.

Design notes
------------
- Subset enumeration is fixed so the ablation is deterministic and
  reproducible without re-invoking Python combinatorics at analysis time:
    * N = 2: all 10 pairs.
    * N = 3: five representative triples. Three of them pin MNLI+RTE (the
      scale-imbalance stressor called out in the methodology) and add each
      of the remaining three tasks in turn; the other two triples vary the
      backdrop while guaranteeing CoLA appears in at least one triple and
      in multiple non-MNLI/RTE contexts.
    * N = 4: all five leave-one-out subsets.
    * N = 5: reused from the restored-head Week 3 argmax rows in
      ``results/best_hparams.json``.
- To keep disk usage bounded, each merged adapter directory is deleted after
  evaluation; only the JSON metrics file per configuration is retained.
- ``--skip-existing`` mirrors the Week 3 convention so partial runs can be
  resumed without reprocessing completed configurations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASKS_ALL: Tuple[str, ...] = ("sst2", "mnli", "qnli", "cola", "rte")
SUBSET_SEP = "__"


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    display_name: str
    merge_script: str
    lambda_value: float
    trim_percentage: int
    b_weight_alpha: float = 0.0
    normalise_a_factors: bool = False
    scale_aware_ties: bool = False


DEFAULT_METHOD_ALIASES: Tuple[str, ...] = (
    "gpa_baseline",
    "gpa_dgpa_saties_wb_0p5",
    "ties",
    "lr_knots",
)
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "gpa_baseline": "GPA+TIES",
    "gpa_dgpa_saties_wb_0p5": "dGPA+saTIES+wB(0.5)",
    "ties": "TIES",
    "lr_knots": "LR-KnOTS+TIES",
}
MERGE_SCRIPT_BY_ALIAS: Dict[str, str] = {
    "gpa_baseline": "merge_gpa_ties.py",
    "gpa_dgpa_saties_wb_0p5": "merge_gpa_ties.py",
    "ties": "merge_ties.py",
    "lr_knots": "merge_lr_knots.py",
}


def all_pairs(tasks: Sequence[str]) -> List[Tuple[str, ...]]:
    return [tuple(sorted(combo)) for combo in itertools.combinations(tasks, 2)]


def representative_triples() -> List[Tuple[str, ...]]:
    # Methodology requirement: MNLI+RTE must appear with each other task to
    # stress scale imbalance, and CoLA must appear in at least one triple.
    # The five selected triples span three MNLI+RTE pairings (covering each
    # of SST-2, QNLI, CoLA as the third task) plus two CoLA-containing
    # triples that break the MNLI/RTE backdrop.
    raw = [
        ("sst2", "mnli", "rte"),
        ("mnli", "qnli", "rte"),
        ("mnli", "cola", "rte"),
        ("sst2", "qnli", "cola"),
        ("mnli", "qnli", "cola"),
    ]
    return [tuple(sorted(triple)) for triple in raw]


def leave_one_out_quartets(tasks: Sequence[str]) -> List[Tuple[str, ...]]:
    # Exactly the five 4-subsets that drop one adapter.
    return [tuple(sorted(combo)) for combo in itertools.combinations(tasks, 4)]


def build_subsets() -> Dict[int, List[Tuple[str, ...]]]:
    return {
        2: all_pairs(TASKS_ALL),
        3: representative_triples(),
        4: leave_one_out_quartets(TASKS_ALL),
    }


def subset_id(subset: Sequence[str]) -> str:
    return SUBSET_SEP.join(subset)


def config_paths(
    *,
    results_root: Path,
    merged_root: Path,
    N: int,
    method: MethodSpec,
    subset: Sequence[str],
) -> Tuple[Path, Path]:
    subset_slug = subset_id(subset)
    results_path = results_root / f"N_{N}" / method.alias / f"{subset_slug}.json"
    merged_dir = merged_root / f"N_{N}" / method.alias / subset_slug
    return results_path, merged_dir


def build_merge_command(
    *,
    python_bin: str,
    method: MethodSpec,
    subset: Sequence[str],
    adapters_dir: Path,
    output_dir: Path,
    results_path: Path,
    majority_sign_method: str,
    eval_batch_size: int,
    max_eval_samples: int | None,
) -> List[str]:
    script_path = PROJECT_ROOT / "scripts" / method.merge_script
    cmd: List[str] = [
        python_bin,
        str(script_path),
        "--adapters-dir",
        str(adapters_dir),
        "--tasks",
        *subset,
        "--trim-percentages",
        str(method.trim_percentage),
        "--lambdas",
        str(method.lambda_value),
        "--output-dir",
        str(output_dir),
        "--results-path",
        str(results_path),
    ]

    if method.merge_script in {"merge_ties.py", "merge_lr_knots.py", "merge_gpa_ties.py"}:
        cmd.extend(["--majority-sign-method", majority_sign_method])

    if method.merge_script == "merge_gpa_ties.py":
        if method.normalise_a_factors:
            cmd.append("--normalise-a-factors")
        if method.scale_aware_ties:
            cmd.append("--scale-aware-ties")
        if method.b_weight_alpha != 0.0:
            cmd.extend(["--b-weight-alpha", str(method.b_weight_alpha)])

    cmd.extend(["--eval-batch-size", str(eval_batch_size)])
    if max_eval_samples is not None:
        cmd.extend(["--max-eval-samples", str(max_eval_samples)])
    return cmd


def run_subprocess(cmd: Sequence[str]) -> None:
    print()
    print(">>> " + " ".join(repr(part) if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def should_skip(results_path: Path) -> bool:
    if not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    runs = payload.get("runs") or []
    if not runs:
        return False
    return isinstance(runs[0].get("evaluation"), dict)


def run_single(
    *,
    python_bin: str,
    method: MethodSpec,
    subset: Sequence[str],
    N: int,
    adapters_dir: Path,
    results_root: Path,
    merged_root: Path,
    majority_sign_method: str,
    eval_batch_size: int,
    max_eval_samples: int | None,
    skip_existing: bool,
    delete_after_eval: bool,
) -> Path:
    results_path, merged_dir = config_paths(
        results_root=results_root,
        merged_root=merged_root,
        N=N,
        method=method,
        subset=subset,
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing and should_skip(results_path):
        print(
            f">>> Skipping N={N} method={method.alias} subset={subset_id(subset)} "
            f"(found existing {results_path})"
        )
        return results_path

    cmd = build_merge_command(
        python_bin=python_bin,
        method=method,
        subset=subset,
        adapters_dir=adapters_dir,
        output_dir=merged_dir,
        results_path=results_path,
        majority_sign_method=majority_sign_method,
        eval_batch_size=eval_batch_size,
        max_eval_samples=max_eval_samples,
    )
    run_subprocess(cmd)

    if delete_after_eval and merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    return results_path


def write_manifest(
    *,
    results_root: Path,
    adapters_dir: Path,
    subsets: Dict[int, List[Tuple[str, ...]]],
    methods: Sequence[MethodSpec],
    best_hparams_path: Path,
    majority_sign_method: str,
    eval_batch_size: int,
    max_eval_samples: int | None,
) -> None:
    manifest = {
        "step": "step_4_1_experiment_11_vary_N",
        "tasks_all": list(TASKS_ALL),
        "adapters_dir": str(adapters_dir),
        "results_root": str(results_root),
        "majority_sign_method": majority_sign_method,
        "eval_batch_size": eval_batch_size,
        "max_eval_samples": max_eval_samples,
        "best_hparams_path": str(best_hparams_path),
        "N_5_source": {
            "source_root": str(best_hparams_path),
            "rationale": (
                "N=5 is sourced from the restored-head Week 3 argmax rows in "
                "results/best_hparams.json so the ablation stays aligned with "
                "the authoritative restored-head operating points."
            ),
        },
        "methods": [
            {
                "alias": method.alias,
                "display_name": method.display_name,
                "merge_script": method.merge_script,
                "lambda": method.lambda_value,
                "trim_percentage": method.trim_percentage,
                "b_weight_alpha": method.b_weight_alpha,
                "normalise_a_factors": method.normalise_a_factors,
                "scale_aware_ties": method.scale_aware_ties,
            }
            for method in methods
        ],
        "subsets_by_N": {
            str(N): [list(subset) for subset in subsets_for_N]
            for N, subsets_for_N in subsets.items()
        },
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used for the merge subprocesses.",
    )
    parser.add_argument(
        "--adapters-dir",
        default=str(PROJECT_ROOT / "adapters"),
        help="Directory with per-task adapter subdirectories (default: seed-42 adapters).",
    )
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "results" / "ablation_N"),
    )
    parser.add_argument(
        "--merged-root",
        default=str(PROJECT_ROOT / "merged_adapters" / "ablation_N"),
        help="Temporary directory for merged adapters (deleted after eval unless --keep-merged).",
    )
    parser.add_argument(
        "--best-hparams-path",
        default=str(PROJECT_ROOT / "results" / "best_hparams.json"),
        help="Path to the restored-head best-hparams summary used for method specs and the N=5 row.",
    )
    parser.add_argument(
        "--n-values",
        default="2,3,4",
        help="Comma-separated N values to run (N=5 is sourced from best_hparams and never rerun here).",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHOD_ALIASES),
        help="Comma-separated method aliases to run.",
    )
    parser.add_argument("--majority-sign-method", choices=["total", "frequency"], default="total")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip configurations whose result JSON already contains an 'evaluation' block (default on).",
    )
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep merged adapter directories after evaluation (default: delete to save disk).",
    )
    return parser


def parse_csv(value: str) -> List[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def resolve_best_hparams_entry(payload: Dict[str, object], alias: str) -> Dict[str, object]:
    methods = payload.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("best_hparams payload is missing a 'methods' block")
    if alias == "gpa_dgpa_saties_wb_0p5":
        gpa_variants = methods.get("gpa_variants", {})
        if not isinstance(gpa_variants, dict) or alias not in gpa_variants:
            raise ValueError(
                "best_hparams payload is missing methods.gpa_variants.gpa_dgpa_saties_wb_0p5"
            )
        entry = gpa_variants[alias]
    else:
        entry = methods.get(alias)
    if not isinstance(entry, dict):
        raise ValueError(f"best_hparams payload is missing methods.{alias}")
    return entry


def load_method_specs(best_hparams_path: Path, aliases: Sequence[str]) -> List[MethodSpec]:
    payload = json.loads(best_hparams_path.read_text(encoding="utf-8"))
    specs: List[MethodSpec] = []
    for alias in aliases:
        if alias not in MERGE_SCRIPT_BY_ALIAS:
            raise ValueError(f"Unsupported method alias '{alias}'")
        entry = resolve_best_hparams_entry(payload, alias)
        hyperparameters = entry.get("hyperparameters", {})
        if not isinstance(hyperparameters, dict):
            raise ValueError(f"Entry for '{alias}' is missing a hyperparameters block")
        lambda_value = hyperparameters.get("lambda")
        trim_percentage = hyperparameters.get("trim_percentage")
        if lambda_value is None or trim_percentage is None:
            raise ValueError(
                f"Entry for '{alias}' must provide both lambda and trim_percentage in best_hparams"
            )
        specs.append(
            MethodSpec(
                alias=alias,
                display_name=str(entry.get("display_name", METHOD_DISPLAY_NAMES[alias])),
                merge_script=MERGE_SCRIPT_BY_ALIAS[alias],
                lambda_value=float(lambda_value),
                trim_percentage=int(trim_percentage),
                b_weight_alpha=float(hyperparameters.get("b_weight_alpha") or 0.0),
                normalise_a_factors=bool(hyperparameters.get("normalise_a_factors") or False),
                scale_aware_ties=bool(hyperparameters.get("scale_aware_ties") or False),
            )
        )
    return specs


def main() -> None:
    args = build_arg_parser().parse_args()

    adapters_dir = Path(args.adapters_dir)
    results_root = Path(args.results_root)
    merged_root = Path(args.merged_root)
    best_hparams_path = Path(args.best_hparams_path)
    n_values = [int(piece) for piece in parse_csv(args.n_values)]
    requested_aliases = parse_csv(args.methods)
    selected_methods = load_method_specs(best_hparams_path, requested_aliases)

    all_subsets = build_subsets()
    subsets_to_run = {N: all_subsets[N] for N in n_values if N in all_subsets}
    skipped_N = [N for N in n_values if N not in all_subsets]
    if skipped_N:
        print(
            f">>> N values {skipped_N} are not runnable here (N=5 is sourced from "
            f"{args.best_hparams_path}); skipping."
        )

    write_manifest(
        results_root=results_root,
        adapters_dir=adapters_dir,
        subsets=all_subsets,
        methods=selected_methods,
        best_hparams_path=best_hparams_path,
        majority_sign_method=args.majority_sign_method,
        eval_batch_size=args.eval_batch_size,
        max_eval_samples=args.max_eval_samples,
    )

    for N, subsets_for_N in subsets_to_run.items():
        for method in selected_methods:
            for subset in subsets_for_N:
                run_single(
                    python_bin=args.python_bin,
                    method=method,
                    subset=subset,
                    N=N,
                    adapters_dir=adapters_dir,
                    results_root=results_root,
                    merged_root=merged_root,
                    majority_sign_method=args.majority_sign_method,
                    eval_batch_size=args.eval_batch_size,
                    max_eval_samples=args.max_eval_samples,
                    skip_existing=args.skip_existing,
                    delete_after_eval=not args.keep_merged,
                )

    print()
    print("Ablation-N merge driver complete.")
    print(f"Per-configuration results under {results_root}")
    print(f"Manifest: {results_root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
