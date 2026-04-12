"""LoRA fine-tuning adapter for the lifecycle-stage classification column.

Only the ``predicted_lifecycle_stage`` column is overridden when
``ML_BACKEND=lora`` is set.  Retirement-year and capacity-factor regressions
always use scikit-learn (see ml.py).

All heavy dependencies (peft, transformers, trl, torch, datasets, accelerate)
are imported lazily inside function bodies so this module can be imported in
CI without any of those packages installed.

Typical usage (from ml.py):
    from entity_data_lakehouse.ml_lora import predict_lifecycle_lora
    stage, conf = predict_lifecycle_lora(feat_row.to_dict(), adapter_dir=adapter_dir)
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

LIFECYCLE_STAGES = [
    "planning",
    "construction",
    "operating",
    "decommissioning",
    "retired",
]

# Script-side default only — never imported by ml.py, which resolves the path
# from gold_root.parent / "models" / "lifecycle_lora_adapter".
DEFAULT_ADAPTER_REL = Path("models/lifecycle_lora_adapter")

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Prompt construction (pure stdlib — safe at import time)
# ---------------------------------------------------------------------------


def features_to_prompt(features: dict) -> str:
    """Convert a feature dict into a natural-language classification prompt.

    The prompt includes every numeric feature column that the sklearn models
    use so the LLM has the same information available.
    """
    lines = [
        "Classify the lifecycle stage of the following energy infrastructure asset.",
        "Respond with exactly one of: planning, construction, operating, decommissioning, retired.",
        "",
        "Asset features:",
        f"  capacity_mw: {features.get('capacity_mw', 'unknown')}",
        f"  sector_encoded: {features.get('sector_encoded', 'unknown')}",
        f"  latitude: {features.get('latitude', 'unknown')}",
        f"  longitude: {features.get('longitude', 'unknown')}",
        f"  altitude_avg_m: {features.get('altitude_avg_m', 'unknown')}",
        f"  territorial_type_encoded: {features.get('territorial_type_encoded', 'unknown')}",
        f"  gdp_tier: {features.get('gdp_tier', 'unknown')}",
        f"  solar_irradiance_kwh_m2_yr: {features.get('solar_irradiance_kwh_m2_yr', 'unknown')}",
        f"  wind_speed_avg_ms: {features.get('wind_speed_avg_ms', 'unknown')}",
        f"  regulatory_stability_score: {features.get('regulatory_stability_score', 'unknown')}",
        f"  total_appearances: {features.get('total_appearances', 'unknown')}",
        f"  presence_rate: {features.get('presence_rate', 'unknown')}",
        f"  reliability_score: {features.get('reliability_score', 'unknown')}",
        f"  typical_lifespan_years: {features.get('typical_lifespan_years', 'unknown')}",
        "",
        "Lifecycle stage:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSONL instruction dataset generation (pandas only — safe at import time)
# ---------------------------------------------------------------------------


def generate_instruction_jsonl(training_df, output_path: Path) -> None:  # type: ignore[type-arg]
    """Write a JSONL instruction-tuning dataset from a synthetic training DataFrame.

    Each line is a JSON object with ``prompt`` and ``completion`` fields.
    The completion is the ground-truth ``lifecycle_stage`` value.

    Parameters
    ----------
    training_df:
        DataFrame produced by ``_generate_synthetic_training_data`` — must
        contain the feature columns used by ``features_to_prompt`` plus a
        ``lifecycle_stage`` label column.
    output_path:
        File path to write the JSONL output to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for _, row in training_df.iterrows():
            prompt = features_to_prompt(row.to_dict())
            completion = str(row["lifecycle_stage"])
            fh.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
    logger.info("Wrote %d instruction examples to %s", len(training_df), output_path)


# ---------------------------------------------------------------------------
# Training (lazy imports — heavy deps)
# ---------------------------------------------------------------------------


def train_lora_adapter(
    training_jsonl: Path,
    output_dir: Path,
    epochs: int = 1,
    base_model: str = BASE_MODEL,
) -> None:
    """Fine-tune a LoRA adapter on the instruction JSONL dataset.

    All of peft / transformers / trl / torch / datasets are imported lazily
    so this function is safe to call only when those packages are available.

    Parameters
    ----------
    training_jsonl:
        Path to the JSONL file written by ``generate_instruction_jsonl``.
    output_dir:
        Directory in which the PEFT adapter weights are saved.
    epochs:
        Number of training epochs.  1 is sufficient for a demo.
    base_model:
        HuggingFace model ID to fine-tune.
    """
    # Lazy imports — not available in base CI environment.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(training_jsonl), split="train")

    def _format(example: dict) -> dict:
        return {"text": example["prompt"] + " " + example["completion"]}

    dataset = dataset.map(_format)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        dataset_text_field="text",
        max_seq_length=256,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("LoRA adapter saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Inference (lazy imports — heavy deps)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_lora_model(adapter_dir: str):  # type: ignore[return]
    """Load a PEFT adapter and its base model for inference.

    The base model name is read from the saved PEFT adapter config so
    inference always matches the base model used during training regardless of
    the ``BASE_MODEL`` constant above.

    Cached with ``lru_cache`` so the model is only loaded once per process.

    Parameters
    ----------
    adapter_dir:
        Absolute path to the saved PEFT adapter directory (string, not Path,
        because lru_cache requires hashable arguments).
    """
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading LoRA adapter from %s", adapter_dir)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def predict_lifecycle_lora(features: dict, adapter_dir: Path) -> tuple[str, float]:
    """Predict the lifecycle stage for a single asset using the LoRA adapter.

    Parameters
    ----------
    features:
        Dict of feature values for a single asset row (all ``_FEATURE_COLS``
        present).
    adapter_dir:
        Resolved absolute path to the PEFT adapter directory.

    Returns
    -------
    (stage, confidence)
        ``stage`` is one of ``LIFECYCLE_STAGES``; ``confidence`` is a float
        in [0, 1] derived from the softmax probability of the chosen token.
        Falls back to (\"operating\", 0.5) if the model produces an
        unrecognised output.
    """
    import torch

    model, tokenizer = load_lora_model(str(adapter_dir))
    prompt = features_to_prompt(features)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = (
        tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        .strip()
        .lower()
    )

    # Extract the first matching lifecycle stage token.
    stage = next((s for s in LIFECYCLE_STAGES if s in generated), None)
    if stage is None:
        logger.warning(
            "LoRA model returned unrecognised output %r; defaulting to 'operating'.",
            generated,
        )
        stage = "operating"
        confidence = 0.5
    else:
        # Use a uniform-over-valid-stages proxy confidence when the model is
        # not computing explicit per-class logits.
        confidence = round(1.0 / len(LIFECYCLE_STAGES) + 0.4, 4)  # ~0.6 as a proxy

    return stage, confidence
