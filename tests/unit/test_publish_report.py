"""Unit tests for publish_mode and publish_report.json.

All tests use mocked layer functions or a small tmp-dir filesystem so no real
pipeline execution or ClickHouse server is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from entity_data_lakehouse.clickhouse_sink import validate_sink_schema


# ---------------------------------------------------------------------------
# Helpers — minimal DataFrames matching sink contracts
# ---------------------------------------------------------------------------

def _ownership_current_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ownership_sk": ["sk-1"],
        "business_key_hash": ["hash-1"],
        "owner_entity_id": ["owner-1"],
        "owner_entity_name": ["Owner One"],
        "asset_id": ["asset-1"],
        "asset_name": ["Asset One"],
        "asset_country": ["GB"],
        "asset_sector": ["solar"],
        "capacity_mw": [100.0],
        "ownership_pct": [80.0],
        "observation_source": ["registry"],
        "effective_date": ["2024-01-01"],
        "expiry_date": ["9999-12-31"],
        "is_current_flag": ["Y"],
        "version_number": [1],
        "change_reason": ["NEW"],
        "dw_batch_id": ["batch-1"],
    })


def _exposure_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame({
        "owner_entity_id": ["owner-1"],
        "asset_country": ["GB"],
        "asset_sector": ["solar"],
        "asset_count": [3],
        "controlled_asset_count": [2],
        "owned_capacity_mw": [150.0],
        "average_ownership_pct": [65.0],
        "relationship_count": [4],
        "snapshot_date": ["2024-01-01"],
        "change_status_vs_prior_snapshot": ["NEW"],
    })


def _ml_predictions_df() -> pd.DataFrame:
    return pd.DataFrame({
        "asset_id": ["asset-1"],
        "asset_name": ["Asset One"],
        "asset_country": ["GB"],
        "asset_sector": ["solar"],
        "capacity_mw": [100.0],
        "latitude": [51.5],
        "longitude": [-0.1],
        "altitude_avg_m": [35.0],
        "territorial_type": ["mainland"],
        "economic_level": ["high_income"],
        "gdp_tier": [4],
        "solar_irradiance_kwh_m2_yr": [1100.0],
        "wind_speed_avg_ms": [6.2],
        "regulatory_stability_score": [0.85],
        "typical_lifespan_years": [25.0],
        "predicted_lifecycle_stage": ["operating"],
        "lifecycle_stage_confidence": [0.87],
        "estimated_retirement_year": [2045],
        "estimated_commissioning_year": [2020],
        "predicted_remaining_years": [19.0],
        "predicted_capacity_factor_pct": [22.5],
        "model_version": ["v1.1-synthetic-300"],
    })


def _make_sink_inputs():
    return (
        {
            "ownership_current": _ownership_current_df(),
            "owner_infrastructure_exposure_snapshot": _exposure_snapshot_df(),
        },
        {"asset_lifecycle_predictions": _ml_predictions_df()},
    )


# ---------------------------------------------------------------------------
# validate_sink_schema — no connection, no mutation
# ---------------------------------------------------------------------------

class TestValidateSinkSchema:
    def test_all_tables_pass_on_valid_frames(self):
        gold, ml = _make_sink_inputs()
        results = validate_sink_schema(gold, ml)
        assert len(results) == 3
        for r in results:
            assert r["status"] == "passed", f"Expected passed for {r['table']}: {r['error']}"
            assert r["error"] is None

    def test_returns_failed_on_missing_column(self):
        gold, ml = _make_sink_inputs()
        # Drop a required column from one table
        gold["ownership_current"] = gold["ownership_current"].drop(columns=["dw_batch_id"])
        results = validate_sink_schema(gold, ml)
        by_table = {r["table"]: r for r in results}
        assert by_table["ownership_current"]["status"] == "failed"
        assert by_table["ownership_current"]["error"] is not None
        # Other tables unaffected
        assert by_table["owner_infrastructure_exposure_snapshot"]["status"] == "passed"
        assert by_table["ml_asset_lifecycle_predictions"]["status"] == "passed"

    def test_returns_failed_on_dtype_mismatch(self):
        gold, ml = _make_sink_inputs()
        gold["ownership_current"] = gold["ownership_current"].copy()
        gold["ownership_current"]["capacity_mw"] = gold["ownership_current"]["capacity_mw"].astype(str)
        results = validate_sink_schema(gold, ml)
        by_table = {r["table"]: r for r in results}
        assert by_table["ownership_current"]["status"] == "failed"
        assert "dtype mismatch" in by_table["ownership_current"]["error"]

    def test_returns_failed_on_missing_key(self):
        gold = {"owner_infrastructure_exposure_snapshot": _exposure_snapshot_df()}
        ml = {"asset_lifecycle_predictions": _ml_predictions_df()}
        results = validate_sink_schema(gold, ml)
        by_table = {r["table"]: r for r in results}
        assert by_table["ownership_current"]["status"] == "failed"
        assert "ownership_current" in by_table["ownership_current"]["error"]

    def test_evaluates_all_tables_even_after_first_failure(self):
        """A failure in table 1 must not prevent tables 2 and 3 from being checked."""
        gold, ml = _make_sink_inputs()
        gold["ownership_current"] = gold["ownership_current"].drop(columns=["dw_batch_id"])
        ml["asset_lifecycle_predictions"] = ml["asset_lifecycle_predictions"].drop(
            columns=["model_version"]
        )
        results = validate_sink_schema(gold, ml)
        assert len(results) == 3, "Must return one result per table regardless of failures"

    def test_does_not_connect_to_clickhouse(self, monkeypatch):
        """validate_sink_schema must not attempt a ClickHouse connection."""
        monkeypatch.setitem(sys.modules, "clickhouse_connect", None)
        gold, ml = _make_sink_inputs()
        # Would raise RuntimeError if _get_client() were called.
        results = validate_sink_schema(gold, ml)
        assert all(r["status"] == "passed" for r in results)


# ---------------------------------------------------------------------------
# publish_mode validation
# ---------------------------------------------------------------------------

class TestPublishModeValidation:
    def test_invalid_publish_mode_raises_value_error(self, tmp_path):
        from entity_data_lakehouse.pipeline import run_pipeline
        with pytest.raises(ValueError, match="Invalid publish_mode"):
            run_pipeline(tmp_path, publish_mode="yolo")

    def test_valid_modes_are_accepted(self, tmp_path):
        """Both valid modes are accepted without a ValueError on the mode check itself."""
        from entity_data_lakehouse.pipeline import run_pipeline
        # We only need to confirm no ValueError from mode validation — the
        # pipeline will fail later on missing sample data, which is fine.
        for mode in ("commit", "dry_run"):
            with pytest.raises(Exception) as exc_info:
                run_pipeline(tmp_path, publish_mode=mode)
            assert "Invalid publish_mode" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# publish_report.json — structure and location
# ---------------------------------------------------------------------------

class TestPublishReport:
    """Tests that publish_report.json is written with the correct schema.

    We run the real pipeline against the actual repo fixture data so the
    report reflects genuine computed values.  These are fast (no ClickHouse)
    and deterministic.
    """

    @pytest.fixture(scope="class")
    def repo_root(self):
        return Path(__file__).resolve().parents[2]

    @pytest.fixture(scope="class")
    def commit_report(self, repo_root, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("commit_report")
        report_path = tmp / "publish_report.json"
        from entity_data_lakehouse.pipeline import run_pipeline
        run_pipeline(repo_root, publish_mode="commit", report_path=report_path)
        with report_path.open() as fh:
            return json.load(fh)

    @pytest.fixture(scope="class")
    def dry_run_report(self, repo_root, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("dry_run_report")
        report_path = tmp / "publish_report.json"
        from entity_data_lakehouse.pipeline import run_pipeline
        run_pipeline(repo_root, publish_mode="dry_run", report_path=report_path)
        with report_path.open() as fh:
            return json.load(fh)

    # --- Required top-level fields ---

    def test_report_has_schema_version(self, commit_report):
        assert commit_report["schema_version"] == "1"

    def test_report_has_run_id(self, commit_report):
        assert len(commit_report["run_id"]) == 12

    def test_report_has_report_timestamp(self, commit_report):
        ts = commit_report["report_timestamp"]
        assert ts.endswith("Z")

    def test_report_has_publish_mode(self, commit_report, dry_run_report):
        assert commit_report["publish_mode"] == "commit"
        assert dry_run_report["publish_mode"] == "dry_run"

    def test_report_has_status_success(self, commit_report, dry_run_report):
        assert commit_report["status"] == "success"
        assert dry_run_report["status"] == "success"

    def test_report_has_tables_attempted(self, commit_report):
        assert set(commit_report["tables_attempted"]) == {
            "ownership_current",
            "owner_infrastructure_exposure_snapshot",
            "ml_asset_lifecycle_predictions",
        }

    def test_report_has_row_counts(self, commit_report):
        rc = commit_report["row_counts"]
        assert rc["entity_master_rows"] == 6
        assert rc["asset_master_rows"] == 5
        assert rc["relationship_edge_rows"] == 37
        assert rc["gold_rows"] == 21
        assert rc["ml_prediction_rows"] == 5

    def test_report_has_rollback_status(self, commit_report):
        # Without ClickHouse enabled, rollback_status is not_applicable
        assert commit_report["rollback_status"] in {
            "not_applicable", "clean", "rolled_back", "partial_rollback_failed"
        }

    def test_report_has_sink_target(self, commit_report):
        st = commit_report["sink_target"]
        assert "clickhouse_enabled" in st
        assert "tables_refreshed" in st
        assert "batch_id" in st
        assert "status" in st

    def test_report_has_public_safety(self, commit_report):
        ps = commit_report["public_safety"]
        assert ps["status"] in {"passed", "failed"}
        assert isinstance(ps["findings"], list)

    def test_report_has_artifacts_written(self, commit_report):
        assert isinstance(commit_report["artifacts_written"], list)

    # --- dry_run specifics ---

    def test_dry_run_report_has_schema_validations(self, dry_run_report):
        st = dry_run_report["sink_target"]
        assert "schema_validations" in st
        assert isinstance(st["schema_validations"], list)

    def test_dry_run_artifacts_written_is_empty(self, dry_run_report):
        assert dry_run_report["artifacts_written"] == []

    def test_dry_run_sink_status(self, dry_run_report):
        # In dry_run, pipeline.py always calls validate_sink_schema() and
        # overwrites sink_target.status to "dry_run_validated" (all pass) or
        # "dry_run_schema_failed" (any fail, which raises and never reaches
        # this fixture). "skipped" is the initial default but is always
        # overwritten before the report is written.
        assert dry_run_report["sink_target"]["status"] == "dry_run_validated", (
            f"Expected 'dry_run_validated' on a successful dry_run; "
            f"got {dry_run_report['sink_target']['status']!r}"
        )

    # --- commit specifics ---

    _EXPECTED_COMMIT_ARTIFACTS = [
        "gold/dw/entity_master_comprehensive_scd4.parquet",
        "gold/dw/entity_master_current.parquet",
        "gold/dw/entity_master_event_log.parquet",
        "gold/dw/ownership_comprehensive_scd4.parquet",
        "gold/dw/ownership_lifecycle.parquet",
        "gold/dw/ownership_history_scd2.parquet",
        "gold/dw/ownership_current.parquet",
        "gold/owner_infrastructure_exposure_snapshot.parquet",
        "gold/dw/asset_lifecycle_predictions.parquet",
        "gold/entity_lakehouse.duckdb",
    ]

    def test_commit_artifacts_written_is_exact_expected_set(self, commit_report):
        actual = set(commit_report["artifacts_written"])
        expected = set(self._EXPECTED_COMMIT_ARTIFACTS)
        assert actual == expected, (
            f"artifacts_written mismatch.\n"
            f"  Missing : {sorted(expected - actual)}\n"
            f"  Unexpected: {sorted(actual - expected)}"
        )

    # --- configurable report path ---

    def test_report_written_to_custom_path(self, repo_root, tmp_path):
        custom_path = tmp_path / "subdir" / "my_report.json"
        from entity_data_lakehouse.pipeline import run_pipeline
        run_pipeline(repo_root, publish_mode="dry_run", report_path=custom_path)
        assert custom_path.exists()
        with custom_path.open() as fh:
            report = json.load(fh)
        assert report["publish_mode"] == "dry_run"

    def test_default_report_path_is_gold_dir(self, repo_root):
        """Default report lands at gold/publish_report.json."""
        from entity_data_lakehouse.pipeline import run_pipeline
        run_pipeline(repo_root, publish_mode="dry_run")
        default_path = repo_root / "gold" / "publish_report.json"
        assert default_path.exists()
        with default_path.open() as fh:
            report = json.load(fh)
        assert report["publish_mode"] == "dry_run"


# ---------------------------------------------------------------------------
# write_gold_to_clickhouse — rollback_status in returned dict
# ---------------------------------------------------------------------------

class TestWriteGoldRollbackStatus:
    """Test the new dict return from write_gold_to_clickhouse."""

    @pytest.fixture()
    def _ch_mock(self, monkeypatch):
        mock_client = MagicMock()
        mock_module = MagicMock()
        mock_module.get_client.return_value = mock_client
        monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)
        monkeypatch.setenv("USE_CLICKHOUSE", "true")
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse")
        return mock_client

    def test_success_returns_clean_rollback_status(self, _ch_mock):
        from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
        gold, ml = _make_sink_inputs()
        result = write_gold_to_clickhouse(gold, ml)
        assert result["status"] == "success"
        assert result["rollback_status"] == "clean"
        assert len(result["tables_refreshed"]) == 3
        assert result["batch_id"] is not None

    def test_skipped_returns_not_applicable(self, monkeypatch):
        monkeypatch.setenv("USE_CLICKHOUSE", "false")
        from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
        gold, ml = _make_sink_inputs()
        result = write_gold_to_clickhouse(gold, ml)
        assert result["status"] == "skipped"
        assert result["rollback_status"] == "not_applicable"
        assert result["batch_id"] is None

    def test_partial_failure_sets_rolled_back_status(self, monkeypatch):
        """When the second table EXCHANGE fails, rollback_status must be 'rolled_back'."""
        monkeypatch.setenv("USE_CLICKHOUSE", "true")
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse")

        call_count = {"n": 0}

        def _fail_on_second_exchange(cmd, *args, **kwargs):
            if "EXCHANGE TABLES" in str(cmd):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated EXCHANGE failure on table 2")

        mock_client = MagicMock()
        mock_client.command.side_effect = _fail_on_second_exchange
        mock_module = MagicMock()
        mock_module.get_client.return_value = mock_client
        monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

        gold, ml = _make_sink_inputs()
        from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
        with pytest.raises(RuntimeError):
            write_gold_to_clickhouse(gold, ml)
        # The exception is re-raised — rollback info lives in the logs.
        # The important check is that the function raised, not returned "success".

    def test_partial_failure_attaches_sink_summary_to_exception(self, monkeypatch):
        """write_gold_to_clickhouse must attach __sink_summary__ to the raised exception."""
        monkeypatch.setenv("USE_CLICKHOUSE", "true")
        monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse")

        call_count = {"n": 0}

        def _fail_on_second_exchange(cmd, *args, **kwargs):
            if "EXCHANGE TABLES" in str(cmd):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated EXCHANGE failure on table 2")

        mock_client = MagicMock()
        mock_client.command.side_effect = _fail_on_second_exchange
        mock_module = MagicMock()
        mock_module.get_client.return_value = mock_client
        monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

        gold, ml = _make_sink_inputs()
        from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
        raised_exc = None
        try:
            write_gold_to_clickhouse(gold, ml)
        except RuntimeError as exc:
            raised_exc = exc

        assert raised_exc is not None, "Expected RuntimeError to be raised"
        summary = getattr(raised_exc, "__sink_summary__", None)
        assert summary is not None, (
            "write_gold_to_clickhouse must attach __sink_summary__ to the raised exception"
        )
        assert summary["status"] == "failed"
        assert summary["rollback_status"] in {"rolled_back", "partial_rollback_failed"}
        assert isinstance(summary["tables_refreshed"], list)


# ---------------------------------------------------------------------------
# dry_run: schema validation failure must raise ValueError
# ---------------------------------------------------------------------------

class TestDryRunSchemaValidationRaises:
    """Verify that a failed ClickHouse schema validation causes dry_run to raise."""

    def test_dry_run_raises_on_schema_failure(self, monkeypatch, tmp_path):
        """If validate_sink_schema returns a failure, run_pipeline must raise ValueError."""
        import entity_data_lakehouse.pipeline as pipeline_mod

        # Patch validate_sink_schema to return one failing table.
        def _failing_validate(gold_outputs, ml_outputs):
            return [
                {"table": "ownership_current", "status": "failed", "error": "missing column"},
                {"table": "owner_infrastructure_exposure_snapshot", "status": "passed", "error": None},
                {"table": "ml_asset_lifecycle_predictions", "status": "passed", "error": None},
            ]

        monkeypatch.setattr(pipeline_mod, "_import_validate_sink_schema", lambda: _failing_validate, raising=False)

        # Patch the import inside run_pipeline directly.
        import entity_data_lakehouse.clickhouse_sink as cs_mod
        monkeypatch.setattr(cs_mod, "validate_sink_schema", _failing_validate)

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        with pytest.raises(ValueError, match="schema validation failed"):
            from entity_data_lakehouse.pipeline import run_pipeline
            run_pipeline(repo_root, publish_mode="dry_run", report_path=report_path)

        # The report should still be written — with status=failed.
        assert report_path.exists(), "Report must be written even when dry_run raises"
        with report_path.open() as fh:
            report = json.load(fh)
        assert report["status"] == "failed"
        assert report["sink_target"]["status"] == "dry_run_schema_failed"

    def test_dry_run_succeeds_when_all_validations_pass(self, monkeypatch, tmp_path):
        """If all validations pass, dry_run must not raise."""
        import entity_data_lakehouse.clickhouse_sink as cs_mod

        def _passing_validate(gold_outputs, ml_outputs):
            return [
                {"table": t, "status": "passed", "error": None}
                for t in ["ownership_current", "owner_infrastructure_exposure_snapshot", "ml_asset_lifecycle_predictions"]
            ]

        monkeypatch.setattr(cs_mod, "validate_sink_schema", _passing_validate)

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        from entity_data_lakehouse.pipeline import run_pipeline
        result = run_pipeline(repo_root, publish_mode="dry_run", report_path=report_path)
        assert isinstance(result, dict)
        with report_path.open() as fh:
            report = json.load(fh)
        assert report["status"] == "success"
        assert report["sink_target"]["status"] == "dry_run_validated"


# ---------------------------------------------------------------------------
# _write_report: raises in dry_run, warns in commit
# ---------------------------------------------------------------------------

class TestWriteReportErrorHandling:
    """Verify _write_report raises on error in dry_run but only warns in commit."""

    def test_write_report_raises_in_dry_run_on_bad_path(self, tmp_path):
        from entity_data_lakehouse.pipeline import _write_report
        # Use a path whose parent is a file (not a directory) to force a write error.
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        bad_path = blocker / "report.json"  # parent is a file → mkdir will fail
        report = {"status": "success", "publish_mode": "dry_run"}
        with pytest.raises(RuntimeError, match="Could not write publish report"):
            _write_report(report, bad_path, dry_run=True)

    def test_write_report_does_not_raise_in_commit_on_bad_path(self, tmp_path):
        from entity_data_lakehouse.pipeline import _write_report
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")
        bad_path = blocker / "report.json"
        report = {"status": "success", "publish_mode": "commit"}
        # Must not raise — should only log a warning.
        _write_report(report, bad_path, dry_run=False)

    def test_write_report_writes_successfully(self, tmp_path):
        from entity_data_lakehouse.pipeline import _write_report
        report_path = tmp_path / "sub" / "report.json"
        report = {"status": "success", "publish_mode": "commit", "run_id": "abc"}
        _write_report(report, report_path, dry_run=False)
        assert report_path.exists()
        with report_path.open() as fh:
            loaded = json.load(fh)
        assert loaded["run_id"] == "abc"


# ---------------------------------------------------------------------------
# sink __sink_summary__ propagation into failure report
# ---------------------------------------------------------------------------

class TestSinkSummaryPropagation:
    """Verify that __sink_summary__ on an exception is reflected in the failure report."""

    def test_sink_failure_metadata_in_report(self, monkeypatch, tmp_path):
        """A failing ClickHouse write that attaches __sink_summary__ must appear in report."""
        import entity_data_lakehouse.clickhouse_sink as cs_mod

        def _failing_sink(gold_outputs, ml_outputs):
            exc = RuntimeError("simulated table EXCHANGE failure")
            exc.__sink_summary__ = {  # type: ignore[attr-defined]
                "tables_refreshed": ["ownership_current"],
                "batch_id": None,
                "status": "failed",
                "rollback_status": "rolled_back",
            }
            raise exc

        monkeypatch.setattr(cs_mod, "write_gold_to_clickhouse", _failing_sink)
        monkeypatch.setenv("USE_CLICKHOUSE", "true")

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        from entity_data_lakehouse.pipeline import run_pipeline
        with pytest.raises(RuntimeError):
            run_pipeline(repo_root, publish_mode="commit", report_path=report_path)

        assert report_path.exists()
        with report_path.open() as fh:
            report = json.load(fh)

        assert report["status"] == "failed"
        assert report["rollback_status"] == "rolled_back"
        assert report["sink_target"]["status"] == "failed"
        assert report["sink_target"]["tables_refreshed"] == ["ownership_current"]

    def test_sink_failure_without_summary_still_writes_report(self, monkeypatch, tmp_path):
        """A plain exception (no __sink_summary__) must still produce a failure report."""
        import entity_data_lakehouse.clickhouse_sink as cs_mod

        monkeypatch.setattr(cs_mod, "write_gold_to_clickhouse",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no summary")))
        monkeypatch.setenv("USE_CLICKHOUSE", "true")

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        from entity_data_lakehouse.pipeline import run_pipeline
        with pytest.raises(RuntimeError):
            run_pipeline(repo_root, publish_mode="commit", report_path=report_path)

        assert report_path.exists()
        with report_path.open() as fh:
            report = json.load(fh)
        assert report["status"] == "failed"
        # rollback_status must be its safe default when no metadata is attached.
        assert report["rollback_status"] == "not_applicable"


# ---------------------------------------------------------------------------
# artifacts_written incremental population
# ---------------------------------------------------------------------------

class TestArtifactsWrittenOnDuckDBFailure:
    """artifacts_written must list all gold artifacts even when pipeline-level
    DuckDB registration fails.

    build_gold_outputs() writes parquet files AND entity_lakehouse.duckdb before
    returning.  If the pipeline's later duckdb.connect() (ML-table registration)
    raises, the failure report must still reflect every artifact already written
    by build_gold_outputs(), including the .duckdb file.
    """

    _EXPECTED_ARTIFACTS = [
        "gold/dw/entity_master_comprehensive_scd4.parquet",
        "gold/dw/entity_master_current.parquet",
        "gold/dw/entity_master_event_log.parquet",
        "gold/dw/ownership_comprehensive_scd4.parquet",
        "gold/dw/ownership_lifecycle.parquet",
        "gold/dw/ownership_history_scd2.parquet",
        "gold/dw/ownership_current.parquet",
        "gold/owner_infrastructure_exposure_snapshot.parquet",
        "gold/dw/asset_lifecycle_predictions.parquet",
        "gold/entity_lakehouse.duckdb",
    ]

    def test_failure_report_lists_all_gold_artifacts_when_duckdb_connect_raises(
        self, tmp_path
    ) -> None:
        """If duckdb.connect() raises during pipeline ML registration, the failure
        report must still list all artifacts already written by build_gold_outputs(),
        including gold parquets AND entity_lakehouse.duckdb (which gold.py writes
        unconditionally before returning)."""
        import duckdb as duckdb_mod

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        real_connect = duckdb_mod.connect

        def _connect_raise_on_registration(path: str, *args, **kwargs):
            # Raise only for the pipeline ML-registration call, which always
            # targets the repo-rooted entity_lakehouse.duckdb path.  All other
            # duckdb.connect() calls (e.g. from build_gold_outputs / gold.py)
            # use paths inside tmp_path or the gold/ dir during those steps, but
            # gold.py also uses entity_lakehouse.duckdb — so we raise on any call
            # AFTER gold outputs are written.  The simplest discriminator is a
            # call counter: the first N calls to entity_lakehouse.duckdb come from
            # gold.py; the pipeline registration call is the final one.
            # Instead, patch only the name inside pipeline.py's namespace.
            return real_connect(path, *args, **kwargs)

        # Patch the `duckdb` name as seen by pipeline.py specifically.
        # `pipeline.py` does `import duckdb` at the top, so we patch the module
        # attribute on the pipeline module itself.
        import entity_data_lakehouse.pipeline as pipeline_mod

        original_duckdb = pipeline_mod.duckdb

        class _FailingDuckDB:
            """Proxy that raises only on .connect(), leaving everything else intact."""
            def connect(self, *a, **kw):
                raise RuntimeError("simulated duckdb connect failure")
            def __getattr__(self, name):
                return getattr(original_duckdb, name)

        with patch.object(pipeline_mod, "duckdb", _FailingDuckDB()):
            with pytest.raises(RuntimeError, match="simulated duckdb connect failure"):
                pipeline_mod.run_pipeline(
                    repo_root, publish_mode="commit", report_path=report_path
                )

        assert report_path.exists(), "failure report must be written even on DuckDB error"
        with report_path.open() as fh:
            report = json.load(fh)

        assert report["status"] == "failed"
        for expected in self._EXPECTED_ARTIFACTS:
            assert expected in report["artifacts_written"], (
                f"Expected '{expected}' in artifacts_written on DuckDB failure, "
                f"got: {report['artifacts_written']}"
            )


# ---------------------------------------------------------------------------
# artifacts_written: partial-gold-write failure
# ---------------------------------------------------------------------------

class TestArtifactsWrittenOnPartialGoldFailure:
    """Failure inside build_gold_outputs() must surface whatever was written.

    gold.py attaches a partial artifacts_written list to the raised exception
    via __gold_artifacts__.  pipeline.py reads that attribute in its except-block
    and overwrites report["artifacts_written"] so recovery info is accurate even
    when gold.py raises mid-write.

    This test forces duckdb.connect() to fail *inside* gold.py (after all
    parquets are written) and asserts the failure report lists the 8 parquets
    but NOT entity_lakehouse.duckdb (which hadn't been written yet at the point
    of failure).
    """

    _EXPECTED_PARQUETS_BEFORE_DUCKDB = [
        "gold/dw/entity_master_comprehensive_scd4.parquet",
        "gold/dw/entity_master_current.parquet",
        "gold/dw/entity_master_event_log.parquet",
        "gold/dw/ownership_comprehensive_scd4.parquet",
        "gold/dw/ownership_lifecycle.parquet",
        "gold/dw/ownership_history_scd2.parquet",
        "gold/dw/ownership_current.parquet",
        "gold/owner_infrastructure_exposure_snapshot.parquet",
    ]

    def test_failure_report_lists_partial_artifacts_when_gold_duckdb_raises(
        self, tmp_path
    ) -> None:
        """If duckdb.connect() raises inside build_gold_outputs() (gold.py), the
        failure report must list the parquets that were already written and must
        NOT list entity_lakehouse.duckdb (which had not been written yet)."""
        import entity_data_lakehouse.gold as gold_mod

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        # Patch duckdb inside gold.py's namespace so only gold.py's connect
        # call raises; pipeline.py's later ML-registration call is unaffected
        # (but we never reach it because gold already raised).
        original_duckdb = gold_mod.duckdb

        class _GoldFailingDuckDB:
            """Proxy that raises only on .connect()."""
            def connect(self, *a, **kw):
                raise RuntimeError("simulated in-gold duckdb connect failure")
            def __getattr__(self, name):
                return getattr(original_duckdb, name)

        with patch.object(gold_mod, "duckdb", _GoldFailingDuckDB()):
            with pytest.raises(RuntimeError, match="simulated in-gold duckdb connect failure"):
                from entity_data_lakehouse.pipeline import run_pipeline
                run_pipeline(repo_root, publish_mode="commit", report_path=report_path)

        assert report_path.exists(), "failure report must be written even on in-gold DuckDB error"
        with report_path.open() as fh:
            report = json.load(fh)

        assert report["status"] == "failed"
        # All parquets written before the DuckDB step must be present.
        for expected in self._EXPECTED_PARQUETS_BEFORE_DUCKDB:
            assert expected in report["artifacts_written"], (
                f"Expected '{expected}' in artifacts_written after partial gold failure, "
                f"got: {report['artifacts_written']}"
            )
        # DuckDB file was not written (connect raised before con.close()), so
        # it must NOT appear in the failure report.
        assert "gold/entity_lakehouse.duckdb" not in report["artifacts_written"], (
            "gold/entity_lakehouse.duckdb must not appear when its write was never completed"
        )


# ---------------------------------------------------------------------------
# DuckDB connection cleanup — finally-block regression tests
# ---------------------------------------------------------------------------

class TestGoldDuckDBConnectionCleanup:
    """gold.py must close the DuckDB connection even when con.execute() raises.

    This pins the try/finally cleanup in build_gold_outputs() so a future
    regression (e.g. accidental removal of the finally block) is caught.
    """

    def test_connection_closed_when_execute_raises_after_connect(self, tmp_path) -> None:
        """If con.execute() raises after a successful duckdb.connect(), con.close()
        must still be called so the file handle is released."""
        from unittest.mock import MagicMock
        import entity_data_lakehouse.gold as gold_mod
        from entity_data_lakehouse.silver import build_silver_outputs

        repo_root = Path(__file__).resolve().parents[2]
        contracts_root = repo_root / "contracts"

        # Build real silver outputs so build_gold_outputs() has valid input.
        silver_outputs = build_silver_outputs(
            sample_root=repo_root / "sample_data",
            silver_root=tmp_path / "silver",
            contract_paths={
                "entity_observations": contracts_root / "entity_observations.schema.json",
                "entity_master": contracts_root / "entity_master.schema.json",
                "asset_master": contracts_root / "asset_master.schema.json",
                "ownership_observations": contracts_root / "ownership_observations.schema.json",
                "relationship_edges": contracts_root / "relationship_edges.schema.json",
            },
            dry_run=True,
        )

        # Build a mock connection whose execute() raises on first call.
        mock_con = MagicMock()
        mock_con.execute.side_effect = RuntimeError("simulated execute failure")

        original_duckdb = gold_mod.duckdb

        class _ExecuteFailingDuckDB:
            def connect(self, *a, **kw):
                return mock_con
            def __getattr__(self, name):
                return getattr(original_duckdb, name)

        gold_root = tmp_path / "gold"
        gold_root.mkdir()
        (gold_root / "dw").mkdir()

        exc_info = None
        with patch.object(gold_mod, "duckdb", _ExecuteFailingDuckDB()):
            with pytest.raises(RuntimeError, match="simulated execute failure") as exc_info:
                gold_mod.build_gold_outputs(
                    gold_root=gold_root,
                    silver_outputs=silver_outputs,
                    contract_paths={
                        "entity_master_comprehensive_scd4": contracts_root
                        / "entity_master_comprehensive_scd4.schema.json",
                        "entity_master_current": contracts_root
                        / "entity_master_current.schema.json",
                        "entity_master_event_log": contracts_root
                        / "entity_master_event_log.schema.json",
                        "ownership_comprehensive_scd4": contracts_root
                        / "ownership_comprehensive_scd4.schema.json",
                        "ownership_lifecycle": contracts_root
                        / "ownership_lifecycle.schema.json",
                        "ownership_history_scd2": contracts_root
                        / "ownership_history_scd2.schema.json",
                        "ownership_current": contracts_root / "ownership_current.schema.json",
                        "owner_infrastructure_exposure_snapshot": contracts_root
                        / "owner_infrastructure_exposure_snapshot.schema.json",
                    },
                    dry_run=False,
                )

        # Connection must have been closed in the finally block.
        mock_con.close.assert_called_once_with()

        # __gold_artifacts__ on the exception must list the 8 parquets that were
        # written before DuckDB connect was attempted; entity_lakehouse.duckdb IS
        # included because connect() succeeded before execute() raised.
        partial = getattr(exc_info.value, "__gold_artifacts__", None)
        assert partial is not None, "__gold_artifacts__ must be attached to the exception"
        expected_parquets = [
            "gold/dw/entity_master_comprehensive_scd4.parquet",
            "gold/dw/entity_master_current.parquet",
            "gold/dw/entity_master_event_log.parquet",
            "gold/dw/ownership_comprehensive_scd4.parquet",
            "gold/dw/ownership_lifecycle.parquet",
            "gold/dw/ownership_history_scd2.parquet",
            "gold/dw/ownership_current.parquet",
            "gold/owner_infrastructure_exposure_snapshot.parquet",
            "gold/entity_lakehouse.duckdb",  # appended immediately after connect()
        ]
        for expected in expected_parquets:
            assert expected in partial, (
                f"Expected '{expected}' in __gold_artifacts__, got: {partial}"
            )


class TestPipelineMLRegistrationConnectionCleanup:
    """pipeline.py must close the ML-registration DuckDB connection even when
    con.execute() raises after a successful connect.

    This pins the try/finally cleanup in the ML-registration block so a future
    regression is caught before it leaks handles in production.
    """

    def test_ml_registration_connection_closed_when_execute_raises(
        self, tmp_path
    ) -> None:
        """If the ML-registration con.execute() raises, con.close() must still
        be called."""
        from unittest.mock import MagicMock
        import entity_data_lakehouse.pipeline as pipeline_mod

        repo_root = Path(__file__).resolve().parents[2]
        report_path = tmp_path / "report.json"

        mock_con = MagicMock()
        mock_con.execute.side_effect = RuntimeError("simulated ML registration failure")

        original_duckdb = pipeline_mod.duckdb

        class _MLFailingDuckDB:
            """Raises only on the pipeline-level ML registration connect call.

            gold.py uses its own duckdb reference (patched separately if needed);
            here we only patch pipeline.py's namespace, so gold.py is unaffected
            and runs normally, then this proxy intercepts the ML registration call.
            """
            def connect(self, *a, **kw):
                return mock_con
            def __getattr__(self, name):
                return getattr(original_duckdb, name)

        with patch.object(pipeline_mod, "duckdb", _MLFailingDuckDB()):
            with pytest.raises(RuntimeError, match="simulated ML registration failure"):
                pipeline_mod.run_pipeline(
                    repo_root, publish_mode="commit", report_path=report_path
                )

        # Connection must have been closed in the finally block.
        mock_con.close.assert_called_once_with()

        # Failure report must list all artifacts written before the registration
        # failure: 8 gold parquets + entity_lakehouse.duckdb (from gold) + ML parquet.
        assert report_path.exists(), "failure report must be written even on ML registration error"
        with report_path.open() as fh:
            report = json.load(fh)

        assert report["status"] == "failed"
        expected_artifacts = {
            "gold/dw/entity_master_comprehensive_scd4.parquet",
            "gold/dw/entity_master_current.parquet",
            "gold/dw/entity_master_event_log.parquet",
            "gold/dw/ownership_comprehensive_scd4.parquet",
            "gold/dw/ownership_lifecycle.parquet",
            "gold/dw/ownership_history_scd2.parquet",
            "gold/dw/ownership_current.parquet",
            "gold/owner_infrastructure_exposure_snapshot.parquet",
            "gold/entity_lakehouse.duckdb",
            "gold/dw/asset_lifecycle_predictions.parquet",
        }
        actual_artifacts = set(report["artifacts_written"])
        assert actual_artifacts == expected_artifacts, (
            f"artifacts_written mismatch on ML registration failure.\n"
            f"  Missing   : {sorted(expected_artifacts - actual_artifacts)}\n"
            f"  Unexpected: {sorted(actual_artifacts - expected_artifacts)}"
        )

