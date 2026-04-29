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
from typing import TYPE_CHECKING

from entity_data_lakehouse.observability import get_langfuse

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Columns from _FEATURE_COLS that are safe to emit as Langfuse telemetry.
#
# Raw geographic coordinates (latitude, longitude, altitude_avg_m) and raw
# capacity figures (capacity_mw) are excluded because they can identify or
# precisely locate physical infrastructure assets even without entity names.
# Only bucketed/encoded/aggregate signals are exported off-box.
#
# This set is intentionally narrow.  If a new feature is added to _FEATURE_COLS
# and should be emitted, it must be explicitly added here with a rationale.
_TELEMETRY_SAFE_COLS: frozenset[str] = frozenset(
    [
        "sector_encoded",           # integer bucket — no location signal
        "territorial_type_encoded", # integer bucket
        "gdp_tier",                 # coarse economic bucket (1-5)
        "solar_irradiance_kwh_m2_yr",  # regional climate average, not pin-point
        "wind_speed_avg_ms",           # regional climate average, not pin-point
        "regulatory_stability_score",  # country-level aggregate
        "total_appearances",        # lifecycle signal — no location
        "presence_rate",            # lifecycle signal
        "reliability_score",        # lifecycle signal
        "typical_lifespan_years",   # sector-level constant
    ]
)

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
# Pin a specific revision for reproducible inference.  Override via env var
# LORA_BASE_MODEL_REVISION to pin a different commit SHA in production.
BASE_MODEL_REVISION = "c1de36e884e19e3e6e5826b56a87a0b83b3d5276"


def validate_adapter_dir(adapter_dir: Path, trusted_root: Path) -> Path:
    """Validate that *adapter_dir* is a real directory inside *trusted_root*.

    Resolves symlinks and rejects:
    - paths that escape the trusted root (including via ``..`` or symlinks)
    - non-directory entries (e.g. a stray file or broken symlink)
    - missing paths

    Parameters
    ----------
    adapter_dir:
        Candidate adapter directory (may be user-supplied via env var).
    trusted_root:
        Root directory that adapter directories must live under
        (typically ``gold_root.parent / "models"``).

    Returns
    -------
    The resolved, validated ``Path``.

    Raises
    ------
    ValueError
        If the path fails any validation check.
    """
    resolved = adapter_dir.resolve()
    root_resolved = trusted_root.resolve()

    if not root_resolved.is_dir():
        raise ValueError(
            f"Trusted adapter root does not exist or is not a directory: {root_resolved}"
        )

    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(
            f"Adapter path {adapter_dir} resolves to {resolved}, which is outside "
            f"the trusted root {root_resolved}.  Refusing to load."
        )

    if not resolved.is_dir():
        raise ValueError(
            f"Adapter path {resolved} is not a directory."
        )

    return resolved


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
    revision: str = BASE_MODEL_REVISION,
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
        HuggingFace model ID to fine-tune.  Must match ``BASE_MODEL`` for
        the inference path to accept the adapter.
    revision:
        Git revision / commit SHA to load.  Must match the revision used
        during inference to ensure reproducible behaviour.  Persisted in
        ``adapter_metadata.json`` alongside the weights.
    """
    # Lazy imports — not available in base CI environment.
    import json as _json
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if base_model != BASE_MODEL:
        raise ValueError(
            f"base_model {base_model!r} does not match pinned BASE_MODEL "
            f"{BASE_MODEL!r}.  Only the pinned model is supported."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s (revision=%s)", base_model, revision)
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        revision=revision,
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

    meta_path = output_dir / "adapter_metadata.json"
    meta_path.write_text(_json.dumps({
        "base_model": base_model,
        "revision": revision,
    }))
    logger.info("LoRA adapter saved to %s (metadata: %s)", output_dir, meta_path)


# ---------------------------------------------------------------------------
# Inference (lazy imports — heavy deps)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_lora_model(adapter_dir: str, revision: str = BASE_MODEL_REVISION):  # type: ignore[return]
    """Load a PEFT adapter and its base model for inference.

    The base model name is read from the saved PEFT adapter config and
    **validated** against the pinned ``BASE_MODEL`` constant.  This prevents
    a tampered adapter from redirecting inference to an arbitrary remote
    model and executing untrusted code.

    Cached with ``lru_cache`` so the model is only loaded once per process.
    The cache key includes both *adapter_dir* and *revision* so changing
    the revision env var between calls is not silently ignored.

    Parameters
    ----------
    adapter_dir:
        Absolute path to the saved PEFT adapter directory (string, not Path,
        because lru_cache requires hashable arguments).
    revision:
        Git revision / commit SHA for the base model.  Defaults to
        ``BASE_MODEL_REVISION``; callers typically read it from the
        ``LORA_BASE_MODEL_REVISION`` environment variable.

    Raises
    ------
    ValueError
        If the adapter config references a base model or revision that does
        not match the pinned ``BASE_MODEL`` / ``BASE_MODEL_REVISION``.
    """
    import json as _json
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading LoRA adapter from %s (revision=%s)", adapter_dir, revision)
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = peft_config.base_model_name_or_path

    if base_model_name != BASE_MODEL:
        raise ValueError(
            f"Adapter base model {base_model_name!r} does not match pinned "
            f"BASE_MODEL {BASE_MODEL!r}.  Refusing to load untrusted adapter."
        )

    # If the adapter was trained with train_lora_adapter(), it will have an
    # adapter_metadata.json recording the revision used.  Validate it matches
    # the inference revision so a stale/updated model cannot silently change
    # behaviour.  Missing or malformed metadata is a hard failure — it means
    # the revision provenance chain is broken and reproducibility cannot be
    # guaranteed.
    meta_path = Path(adapter_dir) / "adapter_metadata.json"
    if not meta_path.exists():
        raise ValueError(
            f"Adapter metadata file not found at {meta_path}.  "
            f"Adapters must be trained with train_lora_adapter() which writes "
            f"revision provenance.  Refusing to load unverified adapter."
        )
    try:
        meta = _json.loads(meta_path.read_text())
    except (_json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Adapter metadata at {meta_path} is corrupt or unreadable: {exc}.  "
            f"Refusing to load adapter with broken provenance."
        )
    trained_rev = meta.get("revision")
    if not trained_rev:
        raise ValueError(
            f"Adapter metadata at {meta_path} has no 'revision' field.  "
            f"Refusing to load adapter with incomplete provenance."
        )
    if trained_rev != revision:
        raise ValueError(
            f"Adapter was trained with revision {trained_rev!r} but "
            f"inference expects {revision!r}.  Refusing to load."
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        revision=revision,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer, base_model_name


def predict_lifecycle_lora(
    features: dict,
    adapter_dir: Path,
    *,
    parent_trace: object | None = None,
) -> tuple[str, float]:
    """Predict the lifecycle stage for a single asset using the LoRA adapter.

    Confidence is computed via **teacher-forced log-probability** over the full
    label string for each of the five valid lifecycle stages:

    1. Tokenise ``prompt + " " + label`` for every candidate label.
    2. Pad and stack all five candidates into a single batched tensor.
    3. Run **one** forward pass for the full batch of 5.
    4. For each candidate, sum the log-probabilities of the label tokens
       (the tokens *after* the pure-prompt prefix).
    5. Apply softmax over the five total log-probs to produce a calibrated
       probability that sums to 1 across the valid label set.

    This approach is robust to multi-token labels and BPE tokenisation
    effects — the full label sequence is scored, not just its first token.

    Parameters
    ----------
    features:
        Dict of feature values for a single asset row (all ``_FEATURE_COLS``
        present).
    adapter_dir:
        Resolved absolute path to the PEFT adapter directory.
    parent_trace:
        Optional Langfuse trace or span object from the calling pipeline run.
        When provided, the row-level generation event is emitted as a child of
        that trace so all LoRA inference events stay correlated with the batch.
        When ``None`` (default), the generation is emitted as a top-level
        observation via the shared Langfuse client.

    Returns
    -------
    (stage, confidence)
        ``stage`` is one of ``LIFECYCLE_STAGES``; ``confidence`` is a float
        in [0, 1] representing the softmax probability of the predicted stage
        over the five valid labels.

    Raises
    ------
    Exception
        Any exception from the forward pass is **re-raised** after logging.
        The caller is responsible for deciding how to handle the failure
        (e.g. keep the upstream sklearn row).  Silently swallowing errors
        here would corrupt output data and misreport model provenance.
    """
    import os as _os
    import torch

    _revision = _os.environ.get("LORA_BASE_MODEL_REVISION", BASE_MODEL_REVISION)
    model, tokenizer, base_model_name = load_lora_model(str(adapter_dir), _revision)
    prompt = features_to_prompt(features)

    # Tokenise the prompt **with** special tokens so we get the exact token
    # sequence the model will see at the start of every candidate.
    prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_ids)

    # Build all 5 candidate sequences upfront so we can run a single batched
    # forward pass instead of 5 serial ones.
    all_label_ids: list[list[int]] = []
    all_full_ids: list[list[int]] = []
    for stage_label in LIFECYCLE_STAGES:
        label_ids = tokenizer.encode(" " + stage_label, add_special_tokens=False)
        all_label_ids.append(label_ids)
        if label_ids:
            all_full_ids.append(prompt_ids + label_ids)
        else:
            # Use a placeholder of minimal length so the batch still has 5
            # entries; the log-prob will be overwritten with -inf below.
            all_full_ids.append(prompt_ids)

    # Pad to the longest sequence in the batch.
    max_len = max(len(ids) for ids in all_full_ids)
    padded = []
    attention_masks = []
    for ids in all_full_ids:
        pad_len = max_len - len(ids)
        padded.append(ids + [tokenizer.pad_token_id or 0] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    input_tensor = torch.tensor(padded)  # (5, max_len)
    attn_tensor = torch.tensor(attention_masks)  # (5, max_len)

    label_log_probs: list[float] = []
    with torch.no_grad():
        forward_out = model(input_tensor, attention_mask=attn_tensor)
        # logits shape: (5, max_len, vocab_size)
        logits = forward_out.logits  # (5, max_len, vocab_size)
        log_softmax = torch.log_softmax(logits, dim=-1)

        for idx, label_ids in enumerate(all_label_ids):
            if not label_ids:
                label_log_probs.append(float("-inf"))
                continue

            # Gather per-token log-probs along the diagonal: for each label
            # token at position (prompt_len + j) in the *unpadded* sequence,
            # the predicting logit sits at index (prompt_len - 1 + j).
            token_lps = log_softmax[
                idx,
                prompt_len - 1: prompt_len - 1 + len(label_ids),
                label_ids,
            ]
            label_lp = float(token_lps.diag().sum().item())
            label_log_probs.append(label_lp)

    lp_tensor = torch.tensor(label_log_probs)
    probs = torch.softmax(lp_tensor, dim=0)
    best_idx = int(torch.argmax(probs).item())
    stage = LIFECYCLE_STAGES[best_idx]
    confidence = round(float(probs[best_idx].item()), 4)

    _emit_lora_chunk(
        1, 1, [stage],
        base_model_name=base_model_name,
        parent_trace=parent_trace,
    )
    return stage, confidence


def predict_lifecycle_lora_batch(
    features_df: "pd.DataFrame",
    adapter_dir: Path,
    *,
    parent_trace: object | None = None,
    chunk_size: int = 32,
) -> list[tuple[str, float] | None]:
    """Predict lifecycle stages for multiple assets using a single LoRA adapter.

    Like :func:`predict_lifecycle_lora` but processes all assets in batched
    chunks, stacking ``(n_assets × 5_labels)`` candidates into one padded
    tensor per chunk instead of running 5 forward passes per asset.

    Rows are converted from the DataFrame chunk-by-chunk to avoid eagerly
    materialising the entire dataset as Python dicts.

    Returns a list the same length as *features_df*.  Each element is
    ``(stage, confidence)`` on success or ``None`` on failure — the caller
    decides how to handle per-asset failures (e.g. keep upstream sklearn).

    Parameters
    ----------
    features_df:
        DataFrame of feature rows (one per asset).
    adapter_dir:
        Resolved path to the PEFT adapter directory.
    parent_trace:
        Optional Langfuse trace/span for correlation.
    chunk_size:
        Maximum number of assets per forward pass.  Controls GPU memory usage.
    """
    n_rows = len(features_df)
    if n_rows == 0:
        return []

    try:
        import os as _os
        import torch

        _revision = _os.environ.get("LORA_BASE_MODEL_REVISION", BASE_MODEL_REVISION)
        model, tokenizer, base_model_name = load_lora_model(str(adapter_dir), _revision)
        pad_id = tokenizer.pad_token_id or 0
    except Exception:
        logger.warning(
            "LoRA setup failed (import or model load); "
            "returning None for all %d rows.",
            n_rows,
            exc_info=True,
        )
        return [None] * n_rows

    # Pre-tokenise all label candidates once (they are the same for every asset).
    label_ids_per_stage: list[list[int]] = []
    for stage_label in LIFECYCLE_STAGES:
        ids = tokenizer.encode(" " + stage_label, add_special_tokens=False)
        label_ids_per_stage.append(ids if ids else [])

    results: list[tuple[str, float] | None] = [None] * n_rows

    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        chunk_df = features_df.iloc[chunk_start:chunk_end]
        # Convert only this chunk to dicts — avoids materialising the full
        # DataFrame upfront.
        chunk_dicts = chunk_df.to_dict("records")

        try:
            # Build all (asset × label) sequences for this chunk.
            all_prompt_lens: list[int] = []
            all_full_ids: list[list[int]] = []
            all_label_ids: list[list[int]] = []
            row_map: list[tuple[int, int]] = []  # (asset_index_in_chunk, label_index)

            for asset_idx, features in enumerate(chunk_dicts):
                prompt = features_to_prompt(features)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                prompt_len = len(prompt_ids)
                all_prompt_lens.append(prompt_len)

                for label_idx, stage_label_ids in enumerate(label_ids_per_stage):
                    if not stage_label_ids:
                        all_label_ids.append([])
                        all_full_ids.append(prompt_ids)
                    else:
                        all_label_ids.append(stage_label_ids)
                        all_full_ids.append(prompt_ids + stage_label_ids)
                    row_map.append((asset_idx, label_idx))

            # Pad to longest sequence in the chunk.
            max_len = max(len(ids) for ids in all_full_ids)
            padded = []
            attention_masks = []
            for ids in all_full_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [pad_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)

            input_tensor = torch.tensor(padded)
            attn_tensor = torch.tensor(attention_masks)

            with torch.no_grad():
                forward_out = model(input_tensor, attention_mask=attn_tensor)
                logits = forward_out.logits
                log_softmax = torch.log_softmax(logits, dim=-1)

            # Score each row in the chunk.
            asset_label_lps: dict[int, list[float]] = {
                i: [] for i in range(len(chunk_dicts))
            }
            for seq_idx, (asset_idx, label_idx) in enumerate(row_map):
                stage_label_ids = all_label_ids[seq_idx]
                if not stage_label_ids:
                    asset_label_lps[asset_idx].append(float("-inf"))
                    continue

                prompt_len = all_prompt_lens[asset_idx]
                token_lps = log_softmax[
                    seq_idx,
                    prompt_len - 1: prompt_len - 1 + len(stage_label_ids),
                    stage_label_ids,
                ]
                label_lp = float(token_lps.diag().sum().item())
                asset_label_lps[asset_idx].append(label_lp)

            # Softmax over 5 labels per asset → pick best.
            chunk_success = 0
            chunk_stages: list[str] = []
            for asset_idx in range(len(chunk_dicts)):
                lps = asset_label_lps[asset_idx]
                lp_tensor = torch.tensor(lps)
                probs = torch.softmax(lp_tensor, dim=0)
                best_idx = int(torch.argmax(probs).item())
                stage = LIFECYCLE_STAGES[best_idx]
                confidence = round(float(probs[best_idx].item()), 4)

                global_idx = chunk_start + asset_idx
                results[global_idx] = (stage, confidence)
                chunk_stages.append(stage)
                chunk_success += 1

            # Emit a single chunk-level telemetry event instead of one per row.
            _emit_lora_chunk(
                chunk_size=len(chunk_dicts),
                chunk_success=chunk_success,
                chunk_stages=chunk_stages,
                base_model_name=base_model_name,
                parent_trace=parent_trace,
            )
        except Exception:
            logger.warning(
                "LoRA batched inference failed for chunk [%d:%d]; "
                "retrying rows individually.",
                chunk_start, chunk_end,
                exc_info=True,
            )
            # Fall back to per-row inference so only genuinely bad rows
            # are marked None; healthy rows in the same chunk are preserved.
            import torch as _torch  # guaranteed available if we got here

            for row_offset, row_features in enumerate(chunk_dicts):
                global_idx = chunk_start + row_offset
                try:
                    prompt = features_to_prompt(row_features)
                    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
                    prompt_len = len(prompt_ids)

                    row_label_lps: list[float] = []
                    with _torch.no_grad():
                        for stage_label_ids in label_ids_per_stage:
                            if not stage_label_ids:
                                row_label_lps.append(float("-inf"))
                                continue
                            full_ids = prompt_ids + stage_label_ids
                            row_tensor = _torch.tensor([full_ids])
                            row_attn = _torch.tensor([[1] * len(full_ids)])
                            out = model(row_tensor, attention_mask=row_attn)
                            row_logits = out.logits
                            row_lsm = _torch.log_softmax(row_logits, dim=-1)
                            tk_lps = row_lsm[
                                0,
                                prompt_len - 1: prompt_len - 1 + len(stage_label_ids),
                                stage_label_ids,
                            ]
                            row_label_lps.append(float(tk_lps.diag().sum().item()))

                    lp_t = _torch.tensor(row_label_lps)
                    p = _torch.softmax(lp_t, dim=0)
                    bi = int(_torch.argmax(p).item())
                    results[global_idx] = (
                        LIFECYCLE_STAGES[bi],
                        round(float(p[bi].item()), 4),
                    )
                except Exception:
                    logger.debug(
                        "LoRA per-row retry failed for global index %d.",
                        global_idx,
                        exc_info=True,
                    )

    return results


def _emit_lora_chunk(
    chunk_size: int,
    chunk_success: int,
    chunk_stages: list[str],
    *,
    base_model_name: str = BASE_MODEL,
    parent_trace: object | None = None,
) -> None:
    """Emit a single Langfuse generation span for a LoRA inference chunk.

    Emits aggregate telemetry (counts, stage distribution) instead of
    per-row events.  This bounds telemetry overhead to O(chunks) rather
    than O(rows) and reduces off-box data exposure.

    No-ops silently when Langfuse is not configured.  Any telemetry failure
    is caught and logged — observability must never abort inference.
    """
    try:
        from collections import Counter

        stage_dist = dict(Counter(chunk_stages))
        generation_kwargs = dict(
            name="lifecycle_lora_batch_chunk",
            input={"chunk_size": chunk_size},
            output={
                "success_count": chunk_success,
                "stage_distribution": stage_dist,
            },
            metadata={
                "model": base_model_name,
                "method": "batched_teacher_forced_log_prob",
            },
        )
        emitter = parent_trace if parent_trace is not None else get_langfuse()
        gen = emitter.generation(**generation_kwargs)
        gen.end()
    except Exception:  # pragma: no cover
        logger.debug(
            "Langfuse chunk emission failed for lifecycle_lora_batch_chunk; ignoring.",
            exc_info=True,
        )
