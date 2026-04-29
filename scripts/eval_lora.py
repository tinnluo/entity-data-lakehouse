"""Evaluate a trained LoRA adapter against the sklearn baseline.

Usage
-----
python scripts/eval_lora.py [--adapter DIR] [--samples N] [--test-split F]

Prints accuracy, per-class F1, and a confusion matrix for both the LoRA
adapter and the sklearn RandomForestClassifier on a held-out synthetic split.

When Langfuse credentials are configured (LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY), each LoRA inference call emits a generation span via
the observability module.

Requires the [lora] optional dependency group:
    pip install -e '.[lora]'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _predict_lora_batch(rows: list[dict], adapter_dir: Path) -> list[str]:
    """Run LoRA inference row-by-row and return predicted stage labels.

    Uses predict_lifecycle_lora (softmax over 5 labels) for each row.
    Langfuse generation spans are emitted automatically by predict_lifecycle_lora
    when credentials are configured.
    """
    predictions = []
    for feat_dict in rows:
        stage, _conf = predict_lifecycle_lora(feat_dict, adapter_dir=adapter_dir)
        predictions.append(stage)
    return predictions


def main() -> None:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    parser = argparse.ArgumentParser(
        description="Evaluate LoRA vs sklearn on lifecycle stage."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=_REPO_ROOT / DEFAULT_ADAPTER_REL,
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Total synthetic samples."
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test fraction (0–1)."
    )
    args = parser.parse_args()

    if not args.adapter.exists():
        print(f"Adapter not found at {args.adapter}. Run train_lora.py first.")
        sys.exit(1)

    lf = get_langfuse()
    trace = lf.trace(name="lora_eval")

    reference_root = _REPO_ROOT / "reference_data"
    country_attrs = _load_country_attributes(reference_root)
    sector_params = _load_sector_lifecycle(reference_root)

    print(f"Generating {args.samples} synthetic samples ...")
    df = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=args.samples,
        seed=99,  # different seed from training
    )

    train_df, test_df = train_test_split(
        df, test_size=args.test_split, random_state=42, stratify=df["lifecycle_stage"]
    )

    # --- sklearn baseline ---
    print("Training sklearn baseline ...")
    models, _ = _train_models(train_df, seed=42)
    X_test = test_df[_FEATURE_COLS].values
    y_true = test_df["lifecycle_stage"].tolist()

    sk_pred_encoded = models["lifecycle_stage_clf"].predict(X_test)
    le2 = LabelEncoder()
    le2.fit(train_df["lifecycle_stage"])
    sk_pred = le2.inverse_transform(sk_pred_encoded)

    sk_acc = accuracy_score(y_true, sk_pred)
    print("\n=== sklearn RandomForest ===")
    print(f"Accuracy: {sk_acc:.3f}")
    print(
        classification_report(y_true, sk_pred, labels=LIFECYCLE_STAGES, zero_division=0)
    )

    # Log sklearn score to Langfuse trace.
    trace.score(name="sklearn_accuracy", value=float(sk_acc))

    # --- LoRA ---
    print(f"\n=== LoRA adapter ({args.adapter}) ===")
    test_rows = [row.to_dict() for _, row in test_df.iterrows()]
    lora_pred = _predict_lora_batch(test_rows, args.adapter)
    lora_acc = accuracy_score(y_true, lora_pred)
    print(f"Accuracy: {lora_acc:.3f}")
    print(
        classification_report(
            y_true, lora_pred, labels=LIFECYCLE_STAGES, zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, lora_pred, labels=LIFECYCLE_STAGES)
    print("Confusion matrix (LoRA):")
    print(f"Labels: {LIFECYCLE_STAGES}")
    print(cm)

    trace.score(name="lora_accuracy", value=float(lora_acc))
    lf.flush()


if __name__ == "__main__":
    main()
