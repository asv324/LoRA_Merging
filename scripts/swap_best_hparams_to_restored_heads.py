"""Produce a single regime-consistent best_hparams.json from the restored-head sweep.

Background
----------
Between 2026-04-17 and 2026-04-21 we discovered that evaluating merged LoRA
adapters without restoring per-task classifier heads collapsed every downstream
metric: CoLA Matthews fell to 0.0, RTE and QNLI to majority-class values, and
SST-2/MNLI to degraded floors. The hyperparameter sweep that produced the
original ``results/best_hparams.json`` had therefore been argmax-ing over a
noisy signal, so every method's operating point was stale.

Step A of Option alpha reran the full 340-configuration Week-3 Step 3.1 grid
under the restored-head evaluation regime. Step B processed those results into
``results/best_hparams_restored_heads.json``.

This script is Step C1 of Option alpha: it produces a single canonical
``results/best_hparams.json`` that holds the restored-head argmaxes for every
non-oracle method, tags each entry with ``evaluation_regime`` so downstream
aggregators can detect the change, and attaches a per-method
``historical_random_head_evaluation`` sub-field so the pre-fix numbers remain
accessible for provenance. The ``task_arithmetic`` entry additionally keeps the
intermediate "restored-head at stale lambda=1.0" snapshot under
``pre_hp_reselection_restored_heads`` so the three-tier provenance trail
(random-head argmax -> restored-head at stale argmax -> restored-head at
re-selected argmax) stays intact.

The oracle row is not regime-dependent (individual adapter eval was always
already loading the correct classifier head via ``PeftModel.from_pretrained``),
so it is copied across verbatim.

Idempotence
-----------
The script reads from ``results/best_hparams_restored_heads.json`` and
``results/best_hparams_historical_random_head.json`` and writes to
``results/best_hparams.json``. Re-running it is safe: it always regenerates the
canonical file from scratch.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESTORED_PATH = RESULTS_DIR / "best_hparams_restored_heads.json"
HISTORICAL_PATH = RESULTS_DIR / "best_hparams_historical_random_head.json"
CANONICAL_PATH = RESULTS_DIR / "best_hparams.json"


RESTORED_REGIME_TAG = "restored_classifier_heads"
RANDOM_HEAD_REGIME_TAG = "random_classifier_heads"


HISTORICAL_NOTE = (
    "Pre-2026-04-21 argmax, selected against evaluations that used a randomly "
    "initialised per-task classifier head instead of the task-specific head "
    "trained during adapter fine-tuning. Under that regime every CoLA score "
    "was forced to 0 and most other tasks regressed to majority-class floors, "
    "so the argmax itself is not directly comparable to the restored-head "
    "primary numbers above. Kept here for provenance only."
)


def _ordered_method_entry(
    entry: Dict[str, Any],
    evaluation_regime: str,
    historical: Dict[str, Any] | None,
    extra_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a copy of ``entry`` with evaluation_regime and provenance attached.

    ``entry`` is re-emitted with keys in a stable, reader-friendly order:
    method identifiers first, then metrics / hyperparameters, then the
    GPA-specific diagnostics, then any provenance sub-fields. Fields that were
    not present in the input are simply skipped.
    """

    source = copy.deepcopy(entry)
    ordered: Dict[str, Any] = {}
    identifier_keys = ("method_key", "display_name", "source_path")
    metric_keys = (
        "average_primary_score",
        "primary_metrics",
        "hyperparameters",
        "variant_label",
        "gpa_convergence",
    )

    for key in identifier_keys:
        if key in source:
            ordered[key] = source.pop(key)

    ordered["evaluation_regime"] = evaluation_regime

    for key in metric_keys:
        if key in source:
            ordered[key] = source.pop(key)

    for key, value in source.items():
        ordered[key] = value

    if extra_provenance:
        for key, value in extra_provenance.items():
            ordered[key] = value

    if historical is not None:
        ordered["historical_random_head_evaluation"] = historical

    return ordered


def _historical_snapshot(
    entry: Dict[str, Any],
    *,
    source_root_hint: str,
) -> Dict[str, Any]:
    """Extract the fields we want to archive from a random-head best entry."""

    snapshot: Dict[str, Any] = {
        "note": HISTORICAL_NOTE,
        "evaluation_regime": RANDOM_HEAD_REGIME_TAG,
        "source_root": source_root_hint,
    }
    for key in (
        "source_path",
        "average_primary_score",
        "primary_metrics",
        "hyperparameters",
        "variant_label",
        "gpa_convergence",
    ):
        if key in entry:
            snapshot[key] = copy.deepcopy(entry[key])
    return snapshot


def _build_task_arithmetic_entry(
    restored_entry: Dict[str, Any],
    historical_current: Dict[str, Any],
    historical_source_root: str,
) -> Dict[str, Any]:
    """Handle the three-tier provenance trail for task_arithmetic."""

    genuine_random_head = historical_current.get("historical_random_head_evaluation")
    if genuine_random_head is None:
        raise RuntimeError(
            "task_arithmetic entry in historical best_hparams file is missing "
            "historical_random_head_evaluation; refusing to synthesise one."
        )

    genuine_random_head = copy.deepcopy(genuine_random_head)
    genuine_random_head.setdefault("evaluation_regime", RANDOM_HEAD_REGIME_TAG)
    genuine_random_head.setdefault("source_root", historical_source_root)
    genuine_random_head.setdefault("note", HISTORICAL_NOTE)

    pre_reselection = {
        "note": (
            "Restored-head evaluation at the prior pinned operating point "
            "(lambda=1.0). Generated by the plain-TA rerun on 2026-04-21 that "
            "grounded the GPA-aligned-TA invariance claim. Superseded by the "
            "primary_metrics above once Step 3.1 was rerun under restored "
            "heads and lambda=0.5 became the honest argmax."
        ),
        "evaluation_regime": RESTORED_REGIME_TAG,
        "source_path": historical_current.get("source_path"),
        "average_primary_score": historical_current.get("average_primary_score"),
        "primary_metrics": copy.deepcopy(historical_current.get("primary_metrics", {})),
        "hyperparameters": copy.deepcopy(historical_current.get("hyperparameters", {})),
    }

    return _ordered_method_entry(
        restored_entry,
        evaluation_regime=RESTORED_REGIME_TAG,
        historical=genuine_random_head,
        extra_provenance={"pre_hp_reselection_restored_heads": pre_reselection},
    )


def _build_gpa_variants(
    restored_variants: Dict[str, Any],
    historical_variants: Dict[str, Any],
    historical_source_root: str,
) -> Dict[str, Any]:
    """Refresh every GPA-family variant, attaching per-variant provenance."""

    rebuilt: Dict[str, Any] = {}
    for variant_key, restored_entry in restored_variants.items():
        historical_entry = historical_variants.get(variant_key)
        historical_snapshot = (
            _historical_snapshot(historical_entry, source_root_hint=historical_source_root)
            if historical_entry is not None
            else None
        )
        rebuilt[variant_key] = _ordered_method_entry(
            restored_entry,
            evaluation_regime=RESTORED_REGIME_TAG,
            historical=historical_snapshot,
        )
    return rebuilt


def _build_canonical(restored: Dict[str, Any], historical: Dict[str, Any]) -> Dict[str, Any]:
    restored_methods = restored.get("methods", {})
    historical_methods = historical.get("methods", {})
    historical_source_root = historical.get("source_root", "")
    restored_source_root = restored.get("source_root", "")

    canonical: Dict[str, Any] = {
        "source_root": restored_source_root,
        "source_root_historical_random_head_evaluation": historical_source_root,
        "evaluation_regime": RESTORED_REGIME_TAG,
        "task_order": restored.get("task_order") or historical.get("task_order", []),
        "selection_metric": restored.get("selection_metric") or historical.get("selection_metric"),
        "swap_metadata": {
            "swapped_at": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "script": "scripts/swap_best_hparams_to_restored_heads.py",
            "notes": (
                "Rebuilt from results/best_hparams_restored_heads.json (Step A/B "
                "of Option alpha). Every non-oracle method now holds the "
                "restored-head argmax; the prior random-head argmax is archived "
                "per-method under historical_random_head_evaluation. Oracle is "
                "regime-independent and copied verbatim."
            ),
        },
        "methods": {},
    }

    out_methods: Dict[str, Any] = canonical["methods"]

    top_level_method_keys = (
        "task_arithmetic",
        "ties",
        "dare_ties",
        "lr_knots",
        "gpa_baseline",
        "gpa_best_enhanced",
    )

    for key in top_level_method_keys:
        if key not in restored_methods:
            raise RuntimeError(f"Restored-heads file is missing method '{key}'")
        restored_entry = restored_methods[key]

        if key == "task_arithmetic":
            if key not in historical_methods:
                raise RuntimeError("Historical file is missing task_arithmetic")
            out_methods[key] = _build_task_arithmetic_entry(
                restored_entry,
                historical_current=historical_methods[key],
                historical_source_root=historical_source_root,
            )
            continue

        historical_entry = historical_methods.get(key)
        historical_snapshot = (
            _historical_snapshot(historical_entry, source_root_hint=historical_source_root)
            if historical_entry is not None
            else None
        )
        out_methods[key] = _ordered_method_entry(
            restored_entry,
            evaluation_regime=RESTORED_REGIME_TAG,
            historical=historical_snapshot,
        )

    restored_gpa_variants = restored_methods.get("gpa_variants", {})
    historical_gpa_variants = historical_methods.get("gpa_variants", {})
    out_methods["gpa_variants"] = _build_gpa_variants(
        restored_gpa_variants,
        historical_gpa_variants,
        historical_source_root=historical_source_root,
    )

    oracle_entry = restored_methods.get("oracle") or historical_methods.get("oracle")
    if oracle_entry is None:
        raise RuntimeError("Neither file has an oracle entry; refusing to continue.")
    oracle_out = copy.deepcopy(oracle_entry)
    oracle_out["evaluation_regime"] = "individual_adapter_eval_regime_independent"
    oracle_out["regime_independence_note"] = (
        "The oracle row is computed from adapters/*/eval_metrics.json, which loads "
        "each adapter with its native classifier head via PeftModel. It is "
        "therefore unaffected by the merge -> eval head restoration change."
    )
    out_methods["oracle"] = oracle_out

    return canonical


def _summary_diff(canonical: Dict[str, Any], historical: Dict[str, Any]) -> str:
    """Return a human-readable summary of how every method moved under the swap."""

    lines = []
    methods_new = canonical["methods"]
    methods_old = historical["methods"]

    def _fmt_hparams(hp: Dict[str, Any]) -> str:
        keep = [
            ("lambda", "lambda"),
            ("trim_percentage", "trim"),
            ("drop_probability", "drop"),
            ("b_weight_alpha", "wB"),
            ("normalise_a_factors", "dA"),
            ("scale_aware_ties", "saTIES"),
        ]
        parts = []
        for key, label in keep:
            if key in hp and hp[key] is not None:
                value = hp[key]
                parts.append(f"{label}={value}")
        return ", ".join(parts) or "-"

    lines.append(
        f"{'method':<34} {'avg (restored)':>16} {'avg (random-head)':>20}   operating point"
    )
    lines.append("-" * 100)
    for key in ("task_arithmetic", "ties", "dare_ties", "lr_knots",
                "gpa_baseline", "gpa_best_enhanced", "oracle"):
        if key not in methods_new:
            continue
        new_entry = methods_new[key]
        old_entry = methods_old.get(key, {})
        new_avg = new_entry.get("average_primary_score")
        old_avg = old_entry.get("average_primary_score")
        new_hp = new_entry.get("hyperparameters", {})
        new_avg_str = f"{new_avg:.4f}" if isinstance(new_avg, float) else "-"
        old_avg_str = f"{old_avg:.4f}" if isinstance(old_avg, float) else "-"
        lines.append(f"{key:<34} {new_avg_str:>16} {old_avg_str:>20}   {_fmt_hparams(new_hp)}")

    return "\n".join(lines)


def main() -> None:
    if not RESTORED_PATH.exists():
        raise SystemExit(f"Missing input: {RESTORED_PATH}")
    if not HISTORICAL_PATH.exists():
        raise SystemExit(f"Missing input: {HISTORICAL_PATH}")

    with RESTORED_PATH.open("r", encoding="utf-8") as f:
        restored = json.load(f)
    with HISTORICAL_PATH.open("r", encoding="utf-8") as f:
        historical = json.load(f)

    canonical = _build_canonical(restored, historical)

    with CANONICAL_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote: {CANONICAL_PATH}")
    print()
    print("Per-method argmax shift (restored-head regime vs random-head regime):")
    print()
    print(_summary_diff(canonical, historical))


if __name__ == "__main__":
    main()
