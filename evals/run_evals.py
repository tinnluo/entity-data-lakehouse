"""Reproducible eval harness: sklearn baseline vs LoRA adapter.

Can be used as an importable module or run directly as a CLI script.

Importable usage::

    from evals.run_evals import run_evals
    report = run_evals()           # writes evals/output/latest_report.json
    print(report["sklearn_accuracy"])

CLI usage::

    python3 evals/run_evals.py [--adapter DIR] [--samples N] [--test-split F]
    make eval

Output
------
``evals/output/latest_report.json`` — machine-readable JSON with the schema::

    {
        "report_timestamp":   "<ISO-8601>",
        "test_samples":       int,
        "sklearn_accuracy":   float,
        "sklearn_f1_per_class": {"planning": float, ...},
        "sklearn_runtime_s":  float,
        "lora_accuracy":      float | null,
        "lora_f1_per_class":  {"planning": float, ...} | null,
        "lora_runtime_s":     float | null,
        "lora_available":     bool,
        "schema_valid":       bool
    }

When no LoRA adapter exists the ``lora_*`` fields are ``null`` and
``lora_available`` is ``false``; the sklearn half still runs and the report
is still written.

When Langfuse credentials are configured (LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY), aggregate accuracy scores are logged to a Langfuse trace.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is importable when executed as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from entity_data_lakehouse.ml import (  # noqa: E402
    _FEATURE_COLS,
    _generate_synthetic_training_data,
    _load_country_attributes,
    _load_sector_lifecycle,
    _train_models,
)
from entity_data_lakehouse.ml_lora import (  # noqa: E402
    DEFAULT_ADAPTER_REL,
    LIFECYCLE_STAGES,
    predict_lifecycle_lora,
)
from entity_data_lakehouse.observability import get_langfuse  # noqa: E402


def run_evals(
    adapter_dir: Path | None = None,
    samples: int = 300,
    test_split: float = 0.2,
    seed_train: int = 42,
    seed_test: int = 99,
    output_path: Path = _REPO_ROOT / "evals" / "output" / "latest_report.json",
) -> dict[str, Any]:
    """Run the full sklearn vs LoRA eval suite and write a JSON report.

    Parameters
    ----------
    adapter_dir:
        Path to the saved PEFT adapter directory.  Defaults to
        ``<repo>/models/lifecycle_lora_adapter``.  If the directory does not
        exist, the LoRA half is skipped and ``lora_available`` is set to
        ``false`` in the report.
    samples:
        Total synthetic sample count used for training + test generation.
    test_split:
        Fraction of ``samples`` reserved as held-out test data.
    seed_train:
        RNG seed used for *training* data generation and model fitting.
    seed_test:
        RNG seed used for *test* data generation (must differ from
        ``seed_train`` to avoid data leakage).
    output_path:
        File path where ``latest_report.json`` is written.

    Returns
    -------
    dict
        The report dictionary (same content as the written JSON file).
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if adapter_dir is None:
        adapter_dir = _REPO_ROOT / DEFAULT_ADAPTER_REL

    reference_root = _REPO_ROOT / "reference_data"
    country_attrs = _load_country_attributes(reference_root)
    sector_params = _load_sector_lifecycle(reference_root)

    # -----------------------------------------------------------------------
    # Generate train / test splits
    # -----------------------------------------------------------------------
    # Training data uses seed_train; test data uses seed_test to avoid leakage.
    train_df = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=samples,
        seed=seed_train,
    )
    test_df_full = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=samples,
        seed=seed_test,
    )
    # Take test_split fraction stratified on lifecycle_stage.
    _, test_df = train_test_split(
        test_df_full,
        test_size=test_split,
        random_state=seed_train,
        stratify=test_df_full["lifecycle_stage"],
    )
    test_df = test_df.reset_index(drop=True)

    y_true = test_df["lifecycle_stage"].tolist()
    X_test = test_df[_FEATURE_COLS].values
    n_test = len(test_df)

    # -----------------------------------------------------------------------
    # sklearn baseline
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    models, _ = _train_models(train_df, seed=seed_train)
    sk_pred_encoded = models["lifecycle_stage_clf"].predict(X_test)
    le = LabelEncoder()
    le.fit(train_df["lifecycle_stage"])
    sk_pred = le.inverse_transform(sk_pred_encoded).tolist()
    sklearn_runtime = round(time.perf_counter() - t0, 4)

    sklearn_accuracy = float(accuracy_score(y_true, sk_pred))
    sklearn_f1 = _per_class_f1(y_true, sk_pred)

    # -----------------------------------------------------------------------
    # Schema validation check (reuse contract JSON from contracts/)
    # -----------------------------------------------------------------------
    schema_valid = _validate_sklearn_predictions(models, train_df, test_df, le)

    # -----------------------------------------------------------------------
    # LoRA evaluation (optional — skipped if adapter absent)
    # -----------------------------------------------------------------------
    lora_available = Path(adapter_dir).exists()
    lora_accuracy: float | None = None
    lora_f1: dict[str, float] | None = None
    lora_runtime: float | None = None

    if lora_available:
        t1 = time.perf_counter()
        lora_pred = [
            predict_lifecycle_lora(row.to_dict(), adapter_dir=Path(adapter_dir))[0]
            for _, row in test_df.iterrows()
        ]
        lora_runtime = round(time.perf_counter() - t1, 4)
        lora_accuracy = float(accuracy_score(y_true, lora_pred))
        lora_f1 = _per_class_f1(y_true, lora_pred)

    # -----------------------------------------------------------------------
    # Assemble report
    # -----------------------------------------------------------------------
    report: dict[str, Any] = {
        "report_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_samples": n_test,
        "sklearn_accuracy": round(sklearn_accuracy, 4),
        "sklearn_f1_per_class": sklearn_f1,
        "sklearn_runtime_s": sklearn_runtime,
        "lora_accuracy": round(lora_accuracy, 4) if lora_accuracy is not None else None,
        "lora_f1_per_class": lora_f1,
        "lora_runtime_s": lora_runtime,
        "lora_available": lora_available,
        "schema_valid": schema_valid,
    }

    # Write report.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    # -----------------------------------------------------------------------
    # Optional Langfuse score logging
    # -----------------------------------------------------------------------
    lf = get_langfuse()
    trace = lf.trace(name="evals_run")
    trace.score(name="sklearn_accuracy", value=sklearn_accuracy)
    if lora_accuracy is not None:
        trace.score(name="lora_accuracy", value=lora_accuracy)
    lf.flush()

    return report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _per_class_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return per-class F1 scores as a dict keyed by stage label."""
    from sklearn.metrics import f1_score

    scores = f1_score(y_true, y_pred, labels=LIFECYCLE_STAGES, average=None, zero_division=0)
    return {stage: round(float(s), 4) for stage, s in zip(LIFECYCLE_STAGES, scores)}


def _validate_sklearn_predictions(models, train_df, test_df, le) -> bool:
    """Check that all predicted stages are valid lifecycle stage labels.

    Returns True if every predicted label is a member of LIFECYCLE_STAGES,
    False otherwise.  This is a lightweight contract check that does not
    require loading the full JSON schema.
    """
    X_test = test_df[_FEATURE_COLS].values
    sk_pred_encoded = models["lifecycle_stage_clf"].predict(X_test)
    sk_pred = le.inverse_transform(sk_pred_encoded).tolist()
    valid_set = set(LIFECYCLE_STAGES)
    return all(s in valid_set for s in sk_pred)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible sklearn vs LoRA eval and write a JSON report."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help=(
            "Path to the saved LoRA adapter directory. "
            f"Defaults to <repo>/{DEFAULT_ADAPTER_REL}."
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Total synthetic samples (default 300)."
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Held-out test fraction (default 0.2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "evals" / "output" / "latest_report.json",
        help="Output path for the JSON report.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    report = run_evals(
        adapter_dir=args.adapter,
        samples=args.samples,
        test_split=args.test_split,
        output_path=args.output,
    )
    print(f"Report written to {args.output}")
    print(f"  sklearn accuracy : {report['sklearn_accuracy']:.4f}")
    if report["lora_available"]:
        print(f"  lora accuracy    : {report['lora_accuracy']:.4f}")
    else:
        print("  lora             : not available (no adapter found)")
    print(f"  schema valid     : {report['schema_valid']}")
