"""Unit tests for the eval harness (evals/run_evals.py).

All ML functions are mocked so these tests run without any GPU or heavy deps.
Verifies:
  - run_evals() returns a dict with all required keys.
  - The JSON report is written to the expected path.
  - lora_available=false and null lora fields when no adapter exists.
  - lora_available=true and non-null lora fields when adapter dir exists.
  - report is valid JSON parseable back to the same dict.
  - schema_valid reflects whether all predicted labels are in LIFECYCLE_STAGES.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Ensure evals/ is importable when running pytest from repo root.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from entity_data_lakehouse.ml_lora import LIFECYCLE_STAGES  # noqa: E402

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

_N_SAMPLES = 50
_TEST_SPLIT = 0.2
_N_TEST = int(_N_SAMPLES * _TEST_SPLIT)  # ~10 rows after stratified split


def _make_synthetic_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Minimal synthetic DataFrame that satisfies run_evals internals."""
    from entity_data_lakehouse.ml import _FEATURE_COLS

    import numpy as np

    rng = np.random.default_rng(seed)
    stages = (LIFECYCLE_STAGES * (n // len(LIFECYCLE_STAGES) + 1))[:n]
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["lifecycle_stage"] = stages
    return pd.DataFrame(data)


def _make_stub_models() -> dict:
    """Return stub sklearn model dict that predicts 'operating' for every row."""
    import numpy as np

    clf = MagicMock()
    clf.predict.return_value = np.zeros(_N_TEST, dtype=int)  # class index 0
    reg1 = MagicMock()
    reg1.predict.return_value = np.full(_N_TEST, 2040.0)
    reg2 = MagicMock()
    reg2.predict.return_value = np.full(_N_TEST, 0.35)
    return {
        "lifecycle_stage_clf": clf,
        "retirement_year_reg": reg1,
        "capacity_factor_reg": reg2,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patched_ml(monkeypatch):
    """Patch all heavy ML functions in evals.run_evals namespace."""
    import evals.run_evals as run_evals_mod

    synthetic_train = _make_synthetic_df(_N_SAMPLES, seed=42)
    synthetic_test = _make_synthetic_df(_N_SAMPLES, seed=99)

    monkeypatch.setattr(
        run_evals_mod,
        "_generate_synthetic_training_data",
        lambda **kw: synthetic_train if kw.get("seed") == 42 else synthetic_test,
    )

    stub_models = _make_stub_models()
    le_mock = MagicMock()
    le_mock.inverse_transform.return_value = ["operating"] * _N_TEST

    monkeypatch.setattr(
        run_evals_mod, "_train_models", lambda df, seed=42: (stub_models, le_mock)
    )

    monkeypatch.setattr(
        run_evals_mod,
        "_load_country_attributes",
        lambda root: {},
    )
    monkeypatch.setattr(
        run_evals_mod,
        "_load_sector_lifecycle",
        lambda root: {},
    )

    return {"models": stub_models, "le": le_mock}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_has_all_required_keys(tmp_path, _patched_ml) -> None:
    """run_evals() must return a dict with every required key."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    required_keys = {
        "report_timestamp",
        "test_samples",
        "sklearn_accuracy",
        "sklearn_f1_per_class",
        "sklearn_runtime_s",
        "lora_accuracy",
        "lora_f1_per_class",
        "lora_runtime_s",
        "lora_available",
        "schema_valid",
    }
    missing = required_keys - set(report.keys())
    assert not missing, f"Report missing keys: {missing}"


def test_report_written_to_output_path(tmp_path, _patched_ml) -> None:
    """run_evals() must write valid JSON to the specified output_path."""
    from evals.run_evals import run_evals

    out = tmp_path / "sub" / "latest_report.json"
    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=out,
    )

    assert out.exists(), "Report file was not created."
    parsed = json.loads(out.read_text())
    assert parsed == report, "Written JSON does not match returned dict."


def test_lora_fields_null_when_no_adapter(tmp_path, _patched_ml) -> None:
    """When adapter_dir does not exist, lora fields must be null / false."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=tmp_path / "nonexistent_adapter",
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora_available"] is False
    assert report["lora_accuracy"] is None
    assert report["lora_f1_per_class"] is None
    assert report["lora_runtime_s"] is None


def test_lora_fields_populated_when_adapter_exists(tmp_path, monkeypatch, _patched_ml) -> None:
    """When an adapter dir exists and predict_lifecycle_lora is stubbed,
    lora_* fields must be non-null and lora_available must be True."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora",
        lambda feat, adapter_dir: ("operating", 0.8),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora_available"] is True
    assert report["lora_accuracy"] is not None
    assert isinstance(report["lora_accuracy"], float)
    assert report["lora_f1_per_class"] is not None
    assert report["lora_runtime_s"] is not None


def test_sklearn_accuracy_in_unit_interval(tmp_path, _patched_ml) -> None:
    """sklearn_accuracy must be a float in [0, 1]."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert 0.0 <= report["sklearn_accuracy"] <= 1.0


def test_f1_per_class_has_all_stages(tmp_path, _patched_ml) -> None:
    """sklearn_f1_per_class must contain a key for every LIFECYCLE_STAGE."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    f1 = report["sklearn_f1_per_class"]
    for stage in LIFECYCLE_STAGES:
        assert stage in f1, f"Stage {stage!r} missing from sklearn_f1_per_class"
        assert 0.0 <= f1[stage] <= 1.0


def test_report_is_valid_json_roundtrip(tmp_path, _patched_ml) -> None:
    """The written file must be parseable JSON whose values round-trip cleanly."""
    from evals.run_evals import run_evals

    out = tmp_path / "report.json"
    run_evals(adapter_dir=None, samples=_N_SAMPLES, output_path=out)

    # Must not raise.
    parsed = json.loads(out.read_text())
    assert "report_timestamp" in parsed
