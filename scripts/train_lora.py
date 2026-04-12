"""Train a LoRA fine-tuned adapter for lifecycle-stage classification.

Usage
-----
python scripts/train_lora.py [--samples N] [--epochs N] [--base-model ID] [--output DIR]

Defaults
--------
--samples 200
--epochs  1
--base-model Qwen/Qwen2.5-0.5B-Instruct
--output models/lifecycle_lora_adapter

The script generates a synthetic instruction dataset using the same
reference data and generation function the main pipeline uses, then fine-tunes
a LoRA adapter on that dataset.

Requires the [lora] optional dependency group:
    pip install -e '.[lora]'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src layout is importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from entity_data_lakehouse.ml import (  # noqa: E402
    _generate_synthetic_training_data,
    _load_country_attributes,
    _load_sector_lifecycle,
)
from entity_data_lakehouse.ml_lora import (  # noqa: E402
    DEFAULT_ADAPTER_REL,
    generate_instruction_jsonl,
    train_lora_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA lifecycle-stage adapter.")
    parser.add_argument(
        "--samples", type=int, default=200, help="Synthetic training samples."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace base model ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / DEFAULT_ADAPTER_REL,
        help="Output directory for the saved adapter.",
    )
    args = parser.parse_args()

    reference_root = _REPO_ROOT / "reference_data"

    print(f"Loading reference data from {reference_root} ...")
    country_attrs = _load_country_attributes(reference_root)
    sector_params = _load_sector_lifecycle(reference_root)

    print(f"Generating {args.samples} synthetic training samples ...")
    training_df = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=args.samples,
        seed=42,
    )

    jsonl_path = args.output.parent / "training_data.jsonl"
    print(f"Writing instruction JSONL to {jsonl_path} ...")
    generate_instruction_jsonl(training_df, jsonl_path)

    print(
        f"Training LoRA adapter ({args.epochs} epoch(s)) on base model {args.base_model} ..."
    )
    train_lora_adapter(
        training_jsonl=jsonl_path,
        output_dir=args.output,
        epochs=args.epochs,
        base_model=args.base_model,
    )
    print(f"Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
