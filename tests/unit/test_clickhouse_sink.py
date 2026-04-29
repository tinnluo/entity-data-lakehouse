"""Unit tests for the ClickHouse analytics sink.

All clickhouse_connect calls are mocked — no running ClickHouse server required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from entity_data_lakehouse.clickhouse_sink import (
    _parse_ddl_columns,
    _prepare_insert_frame,
    _validate_dtypes,
    write_gold_to_clickhouse,
)


# ---------------------------------------------------------------------------
# Minimal test DataFrames matching the exact gold/ML contracts
# ---------------------------------------------------------------------------

def _ownership_current_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
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
        }
    )


def _exposure_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
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
        }
    )


def _ml_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
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
        }
    )


def _make_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    return (
        {
            "ownership_current": _ownership_current_df(),
            "owner_infrastructure_exposure_snapshot": _exposure_snapshot_df(),
        },
        {
            "asset_lifecycle_predictions": _ml_predictions_df(),
        },
    )


# ---------------------------------------------------------------------------
# Shared mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def _ch_mock(monkeypatch):
    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.get_client.return_value = mock_client
    monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse")
    return mock_client


# ---------------------------------------------------------------------------
# No-op path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["false", "False", "FALSE", "", "0"])
def test_noop_when_flag_is_false(monkeypatch, flag) -> None:
    monkeypatch.setenv("USE_CLICKHOUSE", flag)
    gold, ml = _make_inputs()

    import entity_data_lakehouse.clickhouse_sink as mod

    # Temporarily poison clickhouse_connect so any attempted import raises.
    saved = sys.modules.pop("clickhouse_connect", None)
    sys.modules["clickhouse_connect"] = None  # type: ignore[assignment]
    try:
        mod.write_gold_to_clickhouse(gold, ml)  # must not raise
    finally:
        if saved is not None:
            sys.modules["clickhouse_connect"] = saved
        else:
            sys.modules.pop("clickhouse_connect", None)


def test_noop_when_flag_unset(monkeypatch) -> None:
    monkeypatch.delenv("USE_CLICKHOUSE", raising=False)
    gold, ml = _make_inputs()
    sys.modules["clickhouse_connect"] = None  # type: ignore[assignment]
    try:
        from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
        write_gold_to_clickhouse(gold, ml)  # must not raise
    finally:
        sys.modules.pop("clickhouse_connect", None)


# ---------------------------------------------------------------------------
# Active path — database creation
# ---------------------------------------------------------------------------

def test_creates_database(_ch_mock) -> None:
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)
    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    assert "CREATE DATABASE IF NOT EXISTS lakehouse" in commands


# ---------------------------------------------------------------------------
# Active path — atomic refresh sequence
# ---------------------------------------------------------------------------

def test_atomic_refresh_sequence_for_each_table(_ch_mock) -> None:
    """Each table must go through: create live, drop staging, create staging,
    insert into staging, EXCHANGE TABLES (atomic swap), drop staging.
    Staging names include a per-run UUID suffix."""
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]

    for table in (
        "ownership_current",
        "owner_infrastructure_exposure_snapshot",
        "ml_asset_lifecycle_predictions",
    ):
        staging_prefix = f"lakehouse.{table}__staging_"
        # live table DDL
        assert any(f"lakehouse.{table}" in cmd and "CREATE TABLE IF NOT EXISTS" in cmd for cmd in commands)
        # staging cleanup (DROP before create)
        assert any(f"DROP TABLE IF EXISTS {staging_prefix}" in cmd for cmd in commands)
        # staging DDL (CREATE)
        assert any(staging_prefix in cmd and "CREATE TABLE IF NOT EXISTS" in cmd for cmd in commands)
        # atomic exchange using EXCHANGE TABLES
        assert any(
            f"EXCHANGE TABLES lakehouse.{table} AND {staging_prefix}" in cmd
            for cmd in commands
        ), f"EXCHANGE TABLES not found for {table}; commands: {commands}"
        # drop stale staging after exchange
        assert any(f"DROP TABLE IF EXISTS {staging_prefix}" in cmd for cmd in commands)


def test_insert_targets_staging_table(_ch_mock) -> None:
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    insert_targets = {c.args[0] for c in _ch_mock.insert_df.call_args_list}
    # Staging names have a UUID suffix — check prefix only.
    assert any(t.startswith("lakehouse.ownership_current__staging_") for t in insert_targets)
    assert any(t.startswith("lakehouse.owner_infrastructure_exposure_snapshot__staging_") for t in insert_targets)
    assert any(t.startswith("lakehouse.ml_asset_lifecycle_predictions__staging_") for t in insert_targets)


def test_inserts_all_three_tables(_ch_mock) -> None:
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)
    assert _ch_mock.insert_df.call_count == 3


def test_empty_dataframe_skips_insert_but_still_swaps(_ch_mock) -> None:
    """An empty DataFrame must perform the atomic swap (live table is cleared)
    but must not call insert_df for that table."""
    empty = _ownership_current_df().iloc[0:0]
    gold = {
        "ownership_current": empty,
        "owner_infrastructure_exposure_snapshot": _exposure_snapshot_df(),
    }
    ml = {"asset_lifecycle_predictions": _ml_predictions_df()}

    write_gold_to_clickhouse(gold, ml)

    # Only 2 inserts (snapshot + ml); empty ownership_current must not insert.
    assert _ch_mock.insert_df.call_count == 2
    # But the EXCHANGE (atomic swap) must still have happened for ownership_current.
    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    assert any(
        "EXCHANGE TABLES lakehouse.ownership_current AND lakehouse.ownership_current__staging_" in cmd
        for cmd in commands
    )


def test_missing_key_raises_value_error(_ch_mock) -> None:
    gold = {"owner_infrastructure_exposure_snapshot": _exposure_snapshot_df()}
    ml = {"asset_lifecycle_predictions": _ml_predictions_df()}

    with pytest.raises(ValueError, match="expected key 'ownership_current'"):
        write_gold_to_clickhouse(gold, ml)


def test_staging_table_names_are_unique_across_runs(_ch_mock) -> None:
    """Two consecutive runs must use different staging table names."""
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)
    first_targets = {c.args[0] for c in _ch_mock.insert_df.call_args_list}
    _ch_mock.reset_mock()

    write_gold_to_clickhouse(gold, ml)
    second_targets = {c.args[0] for c in _ch_mock.insert_df.call_args_list}

    assert first_targets != second_targets, (
        "Staging table names must differ between runs to prevent concurrent interference"
    )


def test_batch_log_published_after_all_tables(_ch_mock) -> None:
    """After all three tables load, a batch id must be written to lakehouse_batch_log."""
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    assert any("lakehouse_batch_log" in cmd and "CREATE TABLE IF NOT EXISTS" in cmd for cmd in commands)
    assert any("INSERT INTO" in cmd and "lakehouse_batch_log" in cmd for cmd in commands)


def test_batch_log_ddl_uses_run_seq_ordering(_ch_mock) -> None:
    """lakehouse_batch_log DDL must declare run_seq and order by it (not loaded_at)."""
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    create_cmd = next(
        (cmd for cmd in commands if "lakehouse_batch_log" in cmd and "CREATE TABLE IF NOT EXISTS" in cmd),
        None,
    )
    assert create_cmd is not None, "CREATE TABLE for lakehouse_batch_log not found"
    assert "run_seq" in create_cmd, "DDL must declare run_seq column"
    assert "ORDER BY run_seq" in create_cmd, "DDL must order by run_seq, not loaded_at"


def test_batch_log_insert_includes_run_seq(_ch_mock) -> None:
    """INSERT into lakehouse_batch_log must omit run_seq (server evaluates DEFAULT)."""
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    insert_cmd = next(
        (cmd for cmd in commands if "INSERT INTO" in cmd and "lakehouse_batch_log" in cmd),
        None,
    )
    assert insert_cmd is not None, "INSERT into lakehouse_batch_log not found"
    # run_seq is NOT included in the INSERT column list — ClickHouse evaluates
    # toUInt64(now64(9)) server-side via the DEFAULT expression, ensuring a
    # globally consistent ordering key across all pipeline workers.
    assert "run_seq" not in insert_cmd, (
        "INSERT must NOT supply run_seq; let ClickHouse evaluate the DEFAULT server-side"
    )


def test_batch_log_not_published_on_partial_failure(_ch_mock) -> None:
    """If the second table fails, the batch log must NOT be written."""
    gold, ml = _make_inputs()

    call_count = [0]
    original_command = _ch_mock.command.side_effect

    def _fail_on_second_exchange(cmd):
        if "EXCHANGE TABLES" in str(cmd):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("simulated EXCHANGE failure")
        if original_command:
            return original_command(cmd)

    _ch_mock.command.side_effect = _fail_on_second_exchange

    with pytest.raises(RuntimeError):
        write_gold_to_clickhouse(gold, ml)

    commands = [str(c.args[0]) for c in _ch_mock.command.call_args_list]
    assert not any("INSERT INTO" in cmd and "lakehouse_batch_log" in cmd for cmd in commands)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_partial_failure_rolls_back_already_swapped_tables(monkeypatch) -> None:
    """If the second table refresh fails, the first table must be rolled back
    to its previous live data via EXCHANGE TABLES, and batch_log must NOT be
    published — preserving the last-known-good cross-table snapshot."""
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse")

    call_count = {"n": 0}
    exchange_calls: list[str] = []

    def _mock_command(sql, *args, **kwargs):
        sql_str = str(sql)
        if "EXCHANGE TABLES" in sql_str:
            exchange_calls.append(sql_str)
            call_count["n"] += 1
            # Fail on the second EXCHANGE TABLES call (second table refresh).
            if call_count["n"] == 2:
                raise RuntimeError("simulated EXCHANGE TABLES failure on table 2")
        return MagicMock()

    mock_client = MagicMock()
    mock_client.command.side_effect = _mock_command

    mock_module = MagicMock()
    mock_module.get_client.return_value = mock_client
    monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

    gold, ml = _make_inputs()
    with pytest.raises(RuntimeError, match="EXCHANGE TABLES failed"):
        write_gold_to_clickhouse(gold, ml)

    # After failure the first successful EXCHANGE must be reversed.
    # The rollback fires another EXCHANGE on the first table.
    all_commands = [str(c.args[0]) for c in mock_client.command.call_args_list]

    # batch_log must NOT have been published.
    assert not any(
        "INSERT INTO" in cmd and "lakehouse_batch_log" in cmd for cmd in all_commands
    ), "batch_log must not be published after a partial failure"

    # The first table (ownership_current) must have been rolled back:
    # a second EXCHANGE involving ownership_current must appear after the failure.
    ownership_exchanges = [
        cmd for cmd in all_commands
        if "EXCHANGE TABLES" in cmd and "ownership_current" in cmd
    ]
    assert len(ownership_exchanges) >= 2, (
        "First table must be exchanged twice: once for the refresh, once for rollback. "
        f"Got: {ownership_exchanges}"
    )


def test_runtime_error_when_package_missing(monkeypatch) -> None:
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setitem(sys.modules, "clickhouse_connect", None)

    gold, ml = _make_inputs()
    with pytest.raises(RuntimeError, match="clickhouse-connect is not installed"):
        write_gold_to_clickhouse(gold, ml)


def test_runtime_error_on_connection_failure(monkeypatch) -> None:
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "unreachable-host")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "true")  # required for non-local host

    mock_module = MagicMock()
    mock_module.get_client.side_effect = ConnectionRefusedError("refused")
    monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

    gold, ml = _make_inputs()
    with pytest.raises(RuntimeError, match="Failed to connect to ClickHouse"):
        write_gold_to_clickhouse(gold, ml)


def test_invalid_database_identifier_raises(monkeypatch) -> None:
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "lakehouse; DROP DATABASE default")

    gold, ml = _make_inputs()
    with pytest.raises(ValueError, match="Invalid ClickHouse identifier"):
        write_gold_to_clickhouse(gold, ml)


def test_non_local_host_without_tls_raises(monkeypatch) -> None:
    """Plaintext connections to non-localhost hosts must be rejected."""
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse.prod.example.com")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "false")
    monkeypatch.delenv("CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK", raising=False)

    gold, ml = _make_inputs()
    with pytest.raises(ValueError, match="CLICKHOUSE_SECURE must be 'true'"):
        write_gold_to_clickhouse(gold, ml)


def test_escape_hatch_with_public_host_still_raises(monkeypatch) -> None:
    """Even with CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK=true, a public host
    must still be rejected — the flag cannot bypass the TLS guard for
    internet-accessible hosts."""
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse.prod.example.com")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "false")
    monkeypatch.setenv("CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK", "true")

    gold, ml = _make_inputs()
    with pytest.raises(ValueError, match="CLICKHOUSE_SECURE must be 'true'"):
        write_gold_to_clickhouse(gold, ml)


def test_private_network_escape_hatch_allows_compose_service_name(monkeypatch, _ch_mock) -> None:
    """CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK=true with the literal 'clickhouse'
    service name must succeed — this is the standard Docker Compose path."""
    monkeypatch.setenv("CLICKHOUSE_HOST", "clickhouse")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "false")
    monkeypatch.setenv("CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK", "true")
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)  # must not raise


@pytest.mark.parametrize("loopback", ["localhost", "127.0.0.1", "::1"])
def test_loopback_hosts_allowed_without_tls(monkeypatch, loopback, _ch_mock) -> None:
    """Loopback hosts must be allowed without TLS (dev default)."""
    monkeypatch.setenv("CLICKHOUSE_HOST", loopback)
    monkeypatch.setenv("CLICKHOUSE_SECURE", "false")
    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)  # must not raise


def test_secure_flag_passed_to_get_client(monkeypatch) -> None:
    """When CLICKHOUSE_SECURE=true, get_client must receive secure=True."""
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "remote-host")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "true")
    monkeypatch.setenv("CLICKHOUSE_VERIFY", "true")

    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.get_client.return_value = mock_client
    monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    call_kwargs = mock_module.get_client.call_args.kwargs
    assert call_kwargs.get("secure") is True, "secure=True must be passed to get_client"
    assert call_kwargs.get("verify") is True, "verify=True must be passed to get_client"


def test_verify_false_passed_to_get_client(monkeypatch) -> None:
    """CLICKHOUSE_VERIFY=false must set verify=False on get_client."""
    monkeypatch.setenv("USE_CLICKHOUSE", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "remote-host")
    monkeypatch.setenv("CLICKHOUSE_SECURE", "true")
    monkeypatch.setenv("CLICKHOUSE_VERIFY", "false")

    mock_client = MagicMock()
    mock_module = MagicMock()
    mock_module.get_client.return_value = mock_client
    monkeypatch.setitem(sys.modules, "clickhouse_connect", mock_module)

    gold, ml = _make_inputs()
    write_gold_to_clickhouse(gold, ml)

    call_kwargs = mock_module.get_client.call_args.kwargs
    assert call_kwargs.get("verify") is False, "verify=False must be passed to get_client"


# ---------------------------------------------------------------------------
# _prepare_insert_frame — column validation
# ---------------------------------------------------------------------------

def test_prepare_rejects_missing_columns() -> None:
    df = _ownership_current_df().drop(columns=["dw_batch_id"])
    df["batch_id"] = "run-abc"  # batch_id present; dw_batch_id missing
    with pytest.raises(ValueError, match="schema mismatch"):
        _prepare_insert_frame("ownership_current", df)


def test_prepare_rejects_extra_columns() -> None:
    df = _ownership_current_df().copy()
    df["batch_id"] = "run-abc"  # required by DDL
    df["extra_col"] = "oops"    # unexpected — should be rejected
    with pytest.raises(ValueError, match="schema mismatch"):
        _prepare_insert_frame("ownership_current", df)


def test_prepare_returns_projected_copy_in_ddl_order() -> None:
    df = _ml_predictions_df().copy()
    df["batch_id"] = "run-abc"
    shuffled = df[list(reversed(df.columns))]
    result = _prepare_insert_frame("ml_asset_lifecycle_predictions", shuffled)
    assert list(result.columns) == list(
        _parse_ddl_columns("ml_asset_lifecycle_predictions").keys()
    )


def test_prepare_returns_copy_not_view() -> None:
    df = _ownership_current_df().copy()
    df["batch_id"] = "run-abc"
    result = _prepare_insert_frame("ownership_current", df)
    result["ownership_sk"] = "mutated"
    assert df["ownership_sk"].iloc[0] == "sk-1"


# ---------------------------------------------------------------------------
# _validate_dtypes
# ---------------------------------------------------------------------------

def test_validate_dtypes_passes_on_correct_types() -> None:
    df = _ml_predictions_df()
    ddl_cols = _parse_ddl_columns("ml_asset_lifecycle_predictions")
    _validate_dtypes("ml_asset_lifecycle_predictions", df, ddl_cols)  # must not raise


def test_validate_dtypes_accepts_pandas_string_extension_dtype() -> None:
    df = _ownership_current_df().copy()
    df["batch_id"] = "run-1"
    for col in [
        "ownership_sk",
        "business_key_hash",
        "owner_entity_id",
        "owner_entity_name",
        "asset_id",
        "asset_name",
        "asset_country",
        "asset_sector",
        "observation_source",
        "effective_date",
        "expiry_date",
        "is_current_flag",
        "change_reason",
        "dw_batch_id",
        "batch_id",
    ]:
        df[col] = df[col].astype("string")
    ddl_cols = _parse_ddl_columns("ownership_current")
    _validate_dtypes("ownership_current", df, ddl_cols)  # must not raise


def test_validate_dtypes_rejects_string_in_int_column() -> None:
    df = _exposure_snapshot_df().copy()
    df["asset_count"] = df["asset_count"].astype(str)  # object dtype, should be Int64
    ddl_cols = _parse_ddl_columns("owner_infrastructure_exposure_snapshot")
    with pytest.raises(ValueError, match="dtype mismatch"):
        _validate_dtypes("owner_infrastructure_exposure_snapshot", df, ddl_cols)


def test_validate_dtypes_rejects_string_in_float_column() -> None:
    df = _ownership_current_df().copy()
    df["capacity_mw"] = df["capacity_mw"].astype(str)
    ddl_cols = _parse_ddl_columns("ownership_current")
    with pytest.raises(ValueError, match="dtype mismatch"):
        _validate_dtypes("ownership_current", df, ddl_cols)


def test_validate_dtypes_error_lists_all_bad_columns() -> None:
    df = _exposure_snapshot_df().copy()
    df["asset_count"] = df["asset_count"].astype(str)
    df["owned_capacity_mw"] = df["owned_capacity_mw"].astype(str)
    ddl_cols = _parse_ddl_columns("owner_infrastructure_exposure_snapshot")
    with pytest.raises(ValueError) as exc_info:
        _validate_dtypes("owner_infrastructure_exposure_snapshot", df, ddl_cols)
    msg = str(exc_info.value)
    assert "asset_count" in msg
    assert "owned_capacity_mw" in msg


# ---------------------------------------------------------------------------
# _parse_ddl_columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "table_name, expected_cols",
    [
        (
            "ownership_current",
            [
                "ownership_sk", "business_key_hash", "owner_entity_id",
                "owner_entity_name", "asset_id", "asset_name", "asset_country",
                "asset_sector", "capacity_mw", "ownership_pct", "observation_source",
                "effective_date", "expiry_date", "is_current_flag", "version_number",
                "change_reason", "dw_batch_id", "batch_id",
            ],
        ),
        (
            "owner_infrastructure_exposure_snapshot",
            [
                "owner_entity_id", "asset_country", "asset_sector", "asset_count",
                "controlled_asset_count", "owned_capacity_mw", "average_ownership_pct",
                "relationship_count", "snapshot_date", "change_status_vs_prior_snapshot",
                "batch_id",
            ],
        ),
        (
            "ml_asset_lifecycle_predictions",
            [
                "asset_id", "asset_name", "asset_country", "asset_sector",
                "capacity_mw", "latitude", "longitude", "altitude_avg_m",
                "territorial_type", "economic_level", "gdp_tier",
                "solar_irradiance_kwh_m2_yr", "wind_speed_avg_ms",
                "regulatory_stability_score", "typical_lifespan_years",
                "predicted_lifecycle_stage", "lifecycle_stage_confidence",
                "estimated_retirement_year", "estimated_commissioning_year",
                "predicted_remaining_years", "predicted_capacity_factor_pct",
                "model_version", "batch_id",
            ],
        ),
    ],
)
def test_parse_ddl_columns(table_name, expected_cols) -> None:
    assert list(_parse_ddl_columns(table_name).keys()) == expected_cols
