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
    _TELEMETRY_SAFE_COLS,
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


def test_predict_lifecycle_lora_returns_valid_stage(
    tmp_path: Path, monkeypatch
) -> None:
    """predict_lifecycle_lora must return a stage in LIFECYCLE_STAGES and 0 <= conf <= 1.

    The new implementation scores all 5 candidate labels in a single batched
    forward pass using teacher-forced log-probs and applies softmax.  We stub
    torch entirely so the test runs without any heavy deps.

    Strategy: make the "operating" label accumulate a much higher log-prob
    than the others, so after softmax the function must return
    ("operating", high_confidence).
    """
    import math
    import sys

    # Prompt tokens = [1, 2, 3]; each stage gets a single-token label.
    PROMPT_TOKEN_IDS = [1, 2, 3]
    STAGE_TOKEN_IDS = {s: i + 10 for i, s in enumerate(LIFECYCLE_STAGES)}
    HOT_ID = STAGE_TOKEN_IDS["operating"]  # 12

    # ---------------------------------------------------------------------------
    # torch.tensor — returns a list-of-lists wrapper
    # ---------------------------------------------------------------------------
    def _fake_tensor(data, dtype=None):
        return data

    # ---------------------------------------------------------------------------
    # torch.log_softmax on 3-D (batch, seq, vocab) → returns a 3-D wrapper.
    # ---------------------------------------------------------------------------
    def _fake_log_softmax(logits_3d, dim=-1):
        return _FakeLogSoftmax3D(logits_3d)

    class _FakeLogSoftmax3D:
        """Wraps the 3-D logits list; supports [int, slice, list] indexing."""
        def __init__(self, data):
            self._data = data  # list of (list of rows)

        def __getitem__(self, key):
            batch_idx, row_slice, col_ids = key
            rows = self._data[batch_idx][row_slice]
            return _FakeMatrix([[self._row_val(r, c) for c in col_ids] for r in rows])

        @staticmethod
        def _row_val(row, col_id):
            if isinstance(row, dict):
                return row.get(col_id, -5.0)
            return -5.0

    class _FakeMatrix:
        def __init__(self, data):
            self._data = data

        def diag(self):
            n = min(len(self._data), len(self._data[0]) if self._data else 0)
            return _FakeVec([self._data[i][i] for i in range(n)])

    class _FakeVec:
        def __init__(self, vals):
            self._vals = vals

        def sum(self):
            return _FakeScalar(sum(self._vals))

    class _FakeScalar:
        def __init__(self, v):
            self._v = float(v)

        def item(self):
            return self._v

    # ---------------------------------------------------------------------------
    # torch.softmax over the 5 label log-probs.
    # ---------------------------------------------------------------------------
    def _fake_softmax(tensor, dim=0):
        vals = [float(v) for v in tensor]
        max_v = max(vals)
        exps = [math.exp(v - max_v) for v in vals]
        s = sum(exps)
        return [_FakeScalar(e / s) for e in exps]

    def _fake_argmax(tensor):
        vals = [v.item() if hasattr(v, "item") else float(v) for v in tensor]
        m = MagicMock()
        m.item.return_value = vals.index(max(vals))
        return m

    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = lambda s: None
    torch_mock.no_grad.return_value.__exit__ = lambda s, *a: None
    torch_mock.tensor.side_effect = _fake_tensor
    torch_mock.log_softmax.side_effect = _fake_log_softmax
    torch_mock.softmax.side_effect = _fake_softmax
    torch_mock.argmax.side_effect = _fake_argmax

    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    # ---------------------------------------------------------------------------
    # Tokenizer stub
    # ---------------------------------------------------------------------------
    stub_tok = MagicMock()
    stub_tok.pad_token = None

    def _encode(text, add_special_tokens=True):
        if not add_special_tokens:
            for stage in LIFECYCLE_STAGES:
                if text == " " + stage:
                    return [STAGE_TOKEN_IDS[stage]]
            return [99]
        return PROMPT_TOKEN_IDS

    stub_tok.encode.side_effect = _encode

    # ---------------------------------------------------------------------------
    # Model stub: returns 3-D logits (5, seq_len, vocab_size).
    # Each "row" is a dict mapping token_id → logit value.
    # operating's token gets -0.1, everything else gets -5.0.
    # ---------------------------------------------------------------------------
    seq_len = len(PROMPT_TOKEN_IDS) + 1  # +1 for single-token label

    def _make_vocab_row():
        return {tid: (-0.1 if tid == HOT_ID else -5.0) for tid in STAGE_TOKEN_IDS.values()}

    batch_logits = [[_make_vocab_row() for _ in range(seq_len)] for _ in range(len(LIFECYCLE_STAGES))]

    class _FakeForwardOut:
        logits = batch_logits

    stub_model = MagicMock()
    stub_model.return_value = _FakeForwardOut()

    import entity_data_lakehouse.ml_lora as ml_lora_mod
    monkeypatch.setattr(ml_lora_mod, "load_lora_model", lambda _dir, _rev="x": (stub_model, stub_tok, "test-base-model"))

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    stage, conf = predict_lifecycle_lora(_make_fake_features(), adapter_dir=adapter_dir)

    assert stage in LIFECYCLE_STAGES, f"stage {stage!r} not in LIFECYCLE_STAGES"
    assert 0.0 <= conf <= 1.0, f"confidence {conf} out of [0, 1]"
    assert stage == "operating", f"expected 'operating' (highest logit), got {stage!r}"
    assert conf > 0.9, f"expected high confidence for dominant logit, got {conf}"


def test_predict_lifecycle_lora_propagates_forward_pass_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """predict_lifecycle_lora must re-raise exceptions from the forward pass.

    The old behaviour silently returned ("operating", 0.5) on any failure,
    which would corrupt output data and misreport model provenance.  The
    corrected behaviour lets the caller decide how to handle the failure.
    """
    import sys

    torch_mock = MagicMock()
    torch_mock.no_grad.return_value.__enter__ = lambda s: None
    torch_mock.no_grad.return_value.__exit__ = lambda s, *a: None
    monkeypatch.setitem(sys.modules, "torch", torch_mock)

    stub_tok = MagicMock()
    stub_tok.pad_token = None
    stub_tok.encode.return_value = [1, 2, 3]

    stub_model = MagicMock()
    stub_model.side_effect = RuntimeError("simulated forward-pass failure")

    import entity_data_lakehouse.ml_lora as ml_lora_mod
    monkeypatch.setattr(ml_lora_mod, "load_lora_model", lambda _dir, _rev="x": (stub_model, stub_tok, "test-base-model"))

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
        predict_lifecycle_lora(_make_fake_features(), adapter_dir=adapter_dir)


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
        synth.rename(columns={"lifecycle_stage": "_drop"})
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
    """When ML_BACKEND is unset, build_ml_predictions must not import ml_lora."""
    monkeypatch.delenv("ML_BACKEND", raising=False)

    from entity_data_lakehouse.ml import build_ml_predictions

    gold_root = tmp_path / "gold"
    gold_root.mkdir()
    (gold_root / "dw").mkdir()

    import sys

    # Pop ml_lora immediately before the call so we capture only what
    # build_ml_predictions itself imports (not artefacts of other tests).
    sys.modules.pop("entity_data_lakehouse.ml_lora", None)

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
    # ml_lora must NOT have been imported as a side-effect of the sklearn path.
    assert "entity_data_lakehouse.ml_lora" not in sys.modules


def test_build_ml_predictions_lora_override(
    monkeypatch, tmp_path: Path, _pipeline_inputs
) -> None:
    """When ML_BACKEND=lora and adapter exists, the batch stub is called."""
    # Adapter must live under the trusted root (gold_root.parent / "models").
    models_root = tmp_path / "models"
    models_root.mkdir()
    adapter_dir = models_root / "lifecycle_lora_adapter"
    adapter_dir.mkdir()

    monkeypatch.setenv("ML_BACKEND", "lora")
    monkeypatch.setenv("LORA_ADAPTER_PATH", str(adapter_dir))

    call_count = 0

    def _stub_batch(features_df, adapter_dir, **kwargs):
        nonlocal call_count
        call_count += 1
        return [("operating", 0.9) for _ in range(len(features_df))]

    import entity_data_lakehouse.ml_lora as ml_lora_mod

    monkeypatch.setattr(ml_lora_mod, "predict_lifecycle_lora_batch", _stub_batch)

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

    assert call_count == 1, f"Expected 1 batch call, got {call_count}"

    # All lifecycle stages should be 'operating' (what the stub returns).
    assert (predictions["predicted_lifecycle_stage"] == "operating").all()

    # Retirement and capacity columns must not have been mutated by LoRA.
    assert "estimated_retirement_year" in predictions.columns
    assert "predicted_capacity_factor_pct" in predictions.columns
    assert predictions["estimated_retirement_year"].notna().all()
    assert predictions["predicted_capacity_factor_pct"].notna().all()

    # model_version should have '+lora' suffix on every row that succeeded.
    assert predictions["model_version"].str.endswith("+lora").all()


def test_build_ml_predictions_lora_failure_keeps_sklearn_row(
    monkeypatch, tmp_path: Path, _pipeline_inputs
) -> None:
    """When predict_lifecycle_lora_batch returns None, the sklearn row must be
    preserved and model_version must NOT have '+lora' appended."""
    models_root = tmp_path / "models"
    models_root.mkdir()
    adapter_dir = models_root / "lifecycle_lora_adapter"
    adapter_dir.mkdir()

    monkeypatch.setenv("ML_BACKEND", "lora")
    monkeypatch.setenv("LORA_ADAPTER_PATH", str(adapter_dir))

    def _stub_batch_failure(features_df, adapter_dir, **kwargs):
        return [None for _ in range(len(features_df))]

    import entity_data_lakehouse.ml_lora as ml_lora_mod

    monkeypatch.setattr(ml_lora_mod, "predict_lifecycle_lora_batch", _stub_batch_failure)

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
    assert len(predictions) > 0

    # No row should have '+lora' — all failures fell back to sklearn.
    assert not predictions["model_version"].str.endswith("+lora").any(), (
        "Expected no +lora suffix when every LoRA call failed"
    )
    # The stage column must still be valid sklearn output (not silently biased to 'operating').
    assert predictions["predicted_lifecycle_stage"].isin(
        ["planning", "construction", "operating", "decommissioning", "retired"]
    ).all()


# ---------------------------------------------------------------------------
# _TELEMETRY_SAFE_COLS whitelist
# ---------------------------------------------------------------------------


_SENSITIVE_COLS = {"latitude", "longitude", "altitude_avg_m", "capacity_mw"}
_IDENTIFIER_COLS = {"asset_id", "asset_name", "asset_country", "asset_sector"}


def test_telemetry_safe_cols_excludes_sensitive_geo_and_capacity() -> None:
    """Raw lat/lon, altitude, and capacity must not be in the telemetry whitelist."""
    for col in _SENSITIVE_COLS:
        assert col not in _TELEMETRY_SAFE_COLS, (
            f"Sensitive column '{col}' must not be in _TELEMETRY_SAFE_COLS"
        )


def test_telemetry_safe_cols_excludes_identifiers() -> None:
    """Entity identifiers must not appear in the telemetry whitelist."""
    for col in _IDENTIFIER_COLS:
        assert col not in _TELEMETRY_SAFE_COLS, (
            f"Identifier column '{col}' must not be in _TELEMETRY_SAFE_COLS"
        )


def test_telemetry_safe_cols_is_subset_of_feature_cols() -> None:
    """Every column in _TELEMETRY_SAFE_COLS must exist in _FEATURE_COLS."""
    unknown = _TELEMETRY_SAFE_COLS - set(_FEATURE_COLS)
    assert not unknown, (
        f"_TELEMETRY_SAFE_COLS contains columns not in _FEATURE_COLS: {unknown}"
    )


def test_emit_lora_chunk_emits_aggregate_telemetry(monkeypatch) -> None:
    """_emit_lora_chunk must emit chunk-level aggregate data, not per-row features."""
    from entity_data_lakehouse.ml_lora import _emit_lora_chunk

    captured: list[dict] = []

    class _FakeGen:
        def end(self): pass

    class _FakeLF:
        def generation(self, **kwargs):
            captured.append(kwargs)
            return _FakeGen()

    monkeypatch.setattr(
        "entity_data_lakehouse.ml_lora.get_langfuse",
        lambda: _FakeLF(),
    )

    _emit_lora_chunk(
        chunk_size=10,
        chunk_success=8,
        chunk_stages=["operating"] * 5 + ["construction"] * 3,
        base_model_name="my-actual-base-model",
    )

    assert len(captured) == 1
    assert captured[0]["input"] == {"chunk_size": 10}
    assert captured[0]["output"]["success_count"] == 8
    assert captured[0]["output"]["stage_distribution"]["operating"] == 5
    assert captured[0]["output"]["stage_distribution"]["construction"] == 3

    assert captured[0]["metadata"]["model"] == "my-actual-base-model", (
        "Langfuse metadata must emit the resolved base model, not BASE_MODEL constant"
    )

    # No per-row feature data must appear in the telemetry payload.
    for col in _SENSITIVE_COLS | _IDENTIFIER_COLS:
        assert col not in str(captured[0]), f"'{col}' must not appear in chunk telemetry"


def test_emit_lora_chunk_uses_parent_trace_when_provided(monkeypatch) -> None:
    """When parent_trace is given, generation must be emitted on it, not on the global client."""
    from entity_data_lakehouse.ml_lora import _emit_lora_chunk

    parent_captured: list[dict] = []
    global_captured: list[dict] = []

    class _FakeGen:
        def end(self): pass

    class _FakeParentTrace:
        def generation(self, **kwargs):
            parent_captured.append(kwargs)
            return _FakeGen()

    class _FakeGlobalLF:
        def generation(self, **kwargs):
            global_captured.append(kwargs)
            return _FakeGen()

    monkeypatch.setattr(
        "entity_data_lakehouse.ml_lora.get_langfuse",
        lambda: _FakeGlobalLF(),
    )

    parent_trace = _FakeParentTrace()

    _emit_lora_chunk(5, 5, ["operating"] * 5, parent_trace=parent_trace)

    assert len(parent_captured) == 1, "generation must be emitted on parent_trace"
    assert len(global_captured) == 0, "global Langfuse client must NOT be called when parent_trace is set"
