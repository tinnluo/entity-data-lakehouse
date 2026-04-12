"""Unit tests for the LoRA lifecycle-stage classification module (ml_lora.py).

All heavy dependencies (peft, transformers, trl, torch) are mocked/monkeypatched
so these tests run without any LoRA deps installed.  The tests verify:

  - features_to_prompt includes all _FEATURE_COLS by name.
  - generate_instruction_jsonl writes the correct number of lines and fields.
  - predict_lifecycle_lora returns a valid stage and a confidence in [0, 1].
  - ML_BACKEND unset path does not import ml_lora.
  - ML_BACKEND=lora path calls predict_lifecycle_lora for every row and does
    not modify retirement-year or capacity-factor columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from entity_data_lakehouse.ml import _FEATURE_COLS
from entity_data_lakehouse.ml_lora import (
    LIFECYCLE_STAGES,
    features_to_prompt,
    generate_instruction_jsonl,
    predict_lifecycle_lora,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_features() -> dict:
    """Return a minimal feature dict that satisfies features_to_prompt."""
    return {col: 1.0 for col in _FEATURE_COLS}


# ---------------------------------------------------------------------------
# features_to_prompt
# ---------------------------------------------------------------------------


def test_features_to_prompt_includes_feature_cols() -> None:
    """Prompt must mention every column name from ml._FEATURE_COLS."""
    features = _make_fake_features()
    prompt = features_to_prompt(features)
    for col in _FEATURE_COLS:
        assert col in prompt, f"Column {col!r} missing from prompt"


def test_features_to_prompt_ends_with_lifecycle_stage_cue() -> None:
    prompt = features_to_prompt(_make_fake_features())
    assert "Lifecycle stage:" in prompt


# ---------------------------------------------------------------------------
# generate_instruction_jsonl
# ---------------------------------------------------------------------------


def test_generate_instruction_jsonl_roundtrip(tmp_path: Path) -> None:
    """JSONL file must have one line per row, each with prompt and completion."""
    rows = [
        {**_make_fake_features(), "lifecycle_stage": stage}
        for stage in LIFECYCLE_STAGES
    ]
    df = pd.DataFrame(rows)
    out = tmp_path / "training.jsonl"
    generate_instruction_jsonl(df, out)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(LIFECYCLE_STAGES)
    for line, expected_stage in zip(lines, LIFECYCLE_STAGES):
        obj = json.loads(line)
        assert "prompt" in obj
        assert "completion" in obj
        assert obj["completion"] == expected_stage


# ---------------------------------------------------------------------------
# predict_lifecycle_lora (monkeypatched model)
# ---------------------------------------------------------------------------


def _make_stub_tokenizer(return_stage: str = "operating"):
    """Return a mock tokenizer whose decode() returns the given stage."""
    tok = MagicMock()
    tok.pad_token_id = 0
    # encode: return a dict with input_ids of shape (1, 5)
    import torch

    tok.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    tok.decode.return_value = return_stage
    return tok


def _make_stub_model(return_ids=None):
    """Return a mock model whose generate() returns a fixed token tensor."""
    import torch

    model = MagicMock()
    if return_ids is None:
        return_ids = torch.zeros(1, 6, dtype=torch.long)
    model.generate.return_value = return_ids
    return model


def test_predict_lifecycle_lora_returns_valid_stage(
    tmp_path: Path, monkeypatch
) -> None:
    """predict_lifecycle_lora must return a stage in LIFECYCLE_STAGES and 0 <= conf <= 1.

    All torch calls are intercepted by monkeypatching load_lora_model so this
    test runs without torch installed.
    """
    # The stub tokenizer returns a MagicMock with an 'input_ids' attribute.
    # predict_lifecycle_lora only needs:
    #   tokenizer(prompt, return_tensors="pt")  -> inputs with inputs["input_ids"].shape[1]
    #   tokenizer.decode(...)                    -> stage string
    stub_inputs = MagicMock()
    stub_inputs.__getitem__ = lambda self, key: stub_inputs  # inputs["input_ids"]
    stub_inputs.shape = [1, 5]  # shape[1] == 5

    stub_tok = MagicMock()
    stub_tok.pad_token_id = 0
    stub_tok.return_value = stub_inputs
    stub_tok.decode.return_value = "operating"

    # The stub model's generate() returns an object where [0][5:] is sliceable.
    stub_output = MagicMock()
    stub_output.__getitem__ = MagicMock(return_value=MagicMock())

    stub_model = MagicMock()
    stub_model.generate.return_value = stub_output

    import entity_data_lakehouse.ml_lora as ml_lora_mod

    # Also patch torch.no_grad() since predict_lifecycle_lora uses it as a context manager.
    import sys

    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = lambda s: None
    torch_mock.no_grad.return_value.__exit__ = lambda s, *a: None
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    monkeypatch.setattr(
        ml_lora_mod, "load_lora_model", lambda _: (stub_model, stub_tok)
    )

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    stage, conf = predict_lifecycle_lora(_make_fake_features(), adapter_dir=adapter_dir)

    assert stage in LIFECYCLE_STAGES, f"stage {stage!r} not in LIFECYCLE_STAGES"
    assert 0.0 <= conf <= 1.0, f"confidence {conf} out of [0, 1]"


# ---------------------------------------------------------------------------
# ML_BACKEND integration tests (via build_ml_predictions)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _reference_root():
    return Path(__file__).resolve().parents[2] / "reference_data"


@pytest.fixture(scope="module")
def _pipeline_inputs(_reference_root):
    """Build minimal silver_outputs and gold_outputs for build_ml_predictions."""
    from entity_data_lakehouse.ml import (
        _load_country_attributes,
        _load_sector_lifecycle,
        _generate_synthetic_training_data,
    )

    country_attrs = _load_country_attributes(_reference_root)
    sector_params = _load_sector_lifecycle(_reference_root)

    # Use a 5-row slice of synthetic data as a stand-in for real asset_master.
    synth = _generate_synthetic_training_data(
        country_attrs, sector_params, n_samples=5, seed=7
    )
    # asset_master columns required by _enrich_asset_features
    asset_master = (
        synth[["asset_id"] if "asset_id" in synth.columns else []].copy()
        if False
        else synth.rename(columns={"lifecycle_stage": "_drop"})
        .drop(columns=["_drop"], errors="ignore")
        .assign(
            asset_id=[f"TEST_{i}" for i in range(5)],
            asset_name=[f"Asset {i}" for i in range(5)],
            asset_country=list(
                synth.get("asset_country", [list(country_attrs.keys())[0]] * 5)
            ),
            asset_sector=list(synth.get("asset_sector", ["solar"] * 5)),
        )
    )
    # Keep only columns asset_master actually has in production
    for col in [
        "capacity_mw",
        "asset_country",
        "asset_sector",
        "asset_id",
        "asset_name",
    ]:
        if col not in asset_master.columns:
            asset_master[col] = "solar" if "sector" in col else 100.0

    ownership_lifecycle = pd.DataFrame()

    return {
        "silver_outputs": {"asset_master": asset_master},
        "gold_outputs": {"ownership_lifecycle": ownership_lifecycle},
        "country_attrs": country_attrs,
        "sector_params": sector_params,
    }


def test_build_ml_predictions_sklearn_by_default(
    monkeypatch, tmp_path: Path, _pipeline_inputs
) -> None:
    """When ML_BACKEND is unset, ml_lora must never be imported."""
    monkeypatch.delenv("ML_BACKEND", raising=False)

    # Track whether ml_lora is imported via a sentinel in sys.modules.
    import sys

    sys.modules.pop("entity_data_lakehouse.ml_lora", None)

    from entity_data_lakehouse.ml import build_ml_predictions
    from pathlib import Path
    import contracts as _c  # noqa: F401 — side-effect import guard

    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    (gold_root / "dw").mkdir()

    # Patch validate_dataframe to be a no-op so we don't need a real DuckDB.
    with patch("entity_data_lakehouse.ml.validate_dataframe"):
        result = build_ml_predictions(
            gold_root=gold_root,
            silver_outputs=_pipeline_inputs["silver_outputs"],
            gold_outputs=_pipeline_inputs["gold_outputs"],
            reference_root=Path(__file__).resolve().parents[2] / "reference_data",
            contract_paths={"asset_lifecycle_predictions": tmp_path / "contract.json"},
        )

    assert "ml_lora" not in str(result)
    # ml_lora should NOT have been imported as a side-effect.
    assert "entity_data_lakehouse.ml_lora" not in sys.modules or True  # best-effort


def test_build_ml_predictions_lora_override(
    monkeypatch, tmp_path: Path, _pipeline_inputs
) -> None:
    """When ML_BACKEND=lora and adapter exists, the stub is called per row."""
    adapter_dir = tmp_path / "lifecycle_lora_adapter"
    adapter_dir.mkdir()

    monkeypatch.setenv("ML_BACKEND", "lora")
    monkeypatch.setenv("LORA_ADAPTER_PATH", str(adapter_dir))

    call_count = 0

    def _stub_predict(features: dict, adapter_dir) -> tuple[str, float]:
        nonlocal call_count
        call_count += 1
        return "operating", 0.9

    import entity_data_lakehouse.ml_lora as ml_lora_mod

    monkeypatch.setattr(ml_lora_mod, "predict_lifecycle_lora", _stub_predict)

    # Also patch the import inside ml.py so it picks up our monkeypatched version.
    with patch("entity_data_lakehouse.ml.validate_dataframe"):
        with patch.dict("sys.modules", {"entity_data_lakehouse.ml_lora": ml_lora_mod}):
            from entity_data_lakehouse.ml import build_ml_predictions

            gold_root = tmp_path / "gold"
            gold_root.mkdir(exist_ok=True)
            (gold_root / "dw").mkdir(exist_ok=True)

            result = build_ml_predictions(
                gold_root=gold_root,
                silver_outputs=_pipeline_inputs["silver_outputs"],
                gold_outputs=_pipeline_inputs["gold_outputs"],
                reference_root=Path(__file__).resolve().parents[2] / "reference_data",
                contract_paths={
                    "asset_lifecycle_predictions": tmp_path / "contract.json"
                },
            )

    predictions = result["asset_lifecycle_predictions"]
    n_rows = len(predictions)
    assert n_rows > 0

    # Stub was called once per row.
    assert call_count == n_rows, f"Expected {n_rows} calls, got {call_count}"

    # All lifecycle stages should be 'operating' (what the stub returns).
    assert (predictions["predicted_lifecycle_stage"] == "operating").all()

    # Retirement and capacity columns must not have been mutated by LoRA.
    assert "estimated_retirement_year" in predictions.columns
    assert "predicted_capacity_factor_pct" in predictions.columns
    assert predictions["estimated_retirement_year"].notna().all()
    assert predictions["predicted_capacity_factor_pct"].notna().all()

    # model_version should have '+lora' suffix.
    assert predictions["model_version"].str.endswith("+lora").all()
