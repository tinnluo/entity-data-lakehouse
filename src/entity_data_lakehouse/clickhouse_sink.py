"""Optional ClickHouse analytics sink.

When ``USE_CLICKHOUSE=true`` is set, three gold analytics tables are loaded
into ClickHouse after the DuckDB pipeline completes.  When the flag is absent
or ``false``, this module is a no-op and adds no import-time overhead to the
default pipeline path.

Architecture note
-----------------
DuckDB (``gold/entity_lakehouse.duckdb``) is the primary analytics store and
source of truth.  ClickHouse is a **write-through sink** — it receives the
same rows that were already written to DuckDB and is intended for production
OLAP queries.  It is not a backend switch; queries, validation, and failure
handling all centre on DuckDB.

Tables loaded
-------------
- ``ownership_current``             — from gold_outputs["ownership_current"]
- ``owner_infrastructure_exposure_snapshot``
                                    — from gold_outputs["owner_infrastructure_exposure_snapshot"]
- ``ml_asset_lifecycle_predictions``— from ml_outputs["asset_lifecycle_predictions"]

Refresh strategy
----------------
Each sink run performs an **atomic full-refresh** using a staging-table swap:

1. Ensure the live table exists (CREATE TABLE IF NOT EXISTS).
2. Drop any leftover staging table from a previous failed run.
3. Create a staging table (``<table>__staging``) with identical schema.
4. Insert all rows into the staging table.
5. Atomically exchange staging ↔ live with ``EXCHANGE TABLES`` (a single
   atomic operation; readers never see the table missing).
6. Drop the now-stale old-live table (now occupying the staging slot).

If anything fails between steps 3 and 5, the live table still contains the
last successful load.  The staging table is cleaned up on the next run (step 2).

Schema validation
-----------------
The DDL column sets are derived directly from the gold output contracts.
No renaming or default-value fabrication is performed.  The sink:

- validates that the DataFrame has **exactly** the declared columns (missing
  *and* extra columns are rejected)
- validates that each column's pandas dtype is compatible with the declared
  ClickHouse type (e.g. Int64 column must not contain strings)
- projects the DataFrame into DDL declaration order before insert

If any check fails a ``ValueError`` is raised immediately so schema drift is
visible before any data touches ClickHouse.

Connection configuration (environment variables)
-------------------------------------------------
USE_CLICKHOUSE       true | false (default: false)
CLICKHOUSE_HOST      hostname or IP (default: localhost)
CLICKHOUSE_PORT      HTTP port    (default: 8123)
CLICKHOUSE_DATABASE  target database name (default: lakehouse)
CLICKHOUSE_USER      username (default: default)
CLICKHOUSE_PASSWORD  password (default: empty string)

Requires the [clickhouse] optional dependency group::

    pip install -e '.[clickhouse]'

Usage (from pipeline.py)::

    from entity_data_lakehouse.clickhouse_sink import write_gold_to_clickhouse
    write_gold_to_clickhouse(gold_outputs, ml_outputs)
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL templates
#
# Column sets match the actual gold output contracts exactly.
# Derived from the parquet schemas produced by gold.py and ml.py.
# Any change to the upstream gold/ML schema MUST be reflected here.
# ---------------------------------------------------------------------------

_DDL: dict[str, str] = {
    # gold_outputs["ownership_current"]
    # Columns: ownership_sk, business_key_hash, owner_entity_id, owner_entity_name,
    #          asset_id, asset_name, asset_country, asset_sector, capacity_mw,
    #          ownership_pct, observation_source, effective_date, expiry_date,
    #          is_current_flag, version_number, change_reason, dw_batch_id, batch_id
    "ownership_current": """
        CREATE TABLE IF NOT EXISTS {database}.{table} (
            ownership_sk            String,
            business_key_hash       String,
            owner_entity_id         String,
            owner_entity_name       String,
            asset_id                String,
            asset_name              String,
            asset_country           String,
            asset_sector            String,
            capacity_mw             Float64,
            ownership_pct           Float64,
            observation_source      String,
            effective_date          String,
            expiry_date             String,
            is_current_flag         String,
            version_number          Int64,
            change_reason           String,
            dw_batch_id             String,
            batch_id                String
        ) ENGINE = MergeTree()
        ORDER BY (business_key_hash, effective_date)
    """,
    # gold_outputs["owner_infrastructure_exposure_snapshot"]
    # Columns: owner_entity_id, asset_country, asset_sector, asset_count,
    #          controlled_asset_count, owned_capacity_mw, average_ownership_pct,
    #          relationship_count, snapshot_date, change_status_vs_prior_snapshot,
    #          batch_id
    "owner_infrastructure_exposure_snapshot": """
        CREATE TABLE IF NOT EXISTS {database}.{table} (
            owner_entity_id                 String,
            asset_country                   String,
            asset_sector                    String,
            asset_count                     Int64,
            controlled_asset_count          Int64,
            owned_capacity_mw               Float64,
            average_ownership_pct           Float64,
            relationship_count              Int64,
            snapshot_date                   String,
            change_status_vs_prior_snapshot String,
            batch_id                        String
        ) ENGINE = MergeTree()
        ORDER BY (owner_entity_id, snapshot_date, asset_country, asset_sector)
    """,
    # ml_outputs["asset_lifecycle_predictions"]
    # Columns: asset_id, asset_name, asset_country, asset_sector, capacity_mw,
    #          latitude, longitude, altitude_avg_m, territorial_type, economic_level,
    #          gdp_tier, solar_irradiance_kwh_m2_yr, wind_speed_avg_ms,
    #          regulatory_stability_score, typical_lifespan_years,
    #          predicted_lifecycle_stage, lifecycle_stage_confidence,
    #          estimated_retirement_year, estimated_commissioning_year,
    #          predicted_remaining_years, predicted_capacity_factor_pct,
    #          model_version, batch_id
    "ml_asset_lifecycle_predictions": """
        CREATE TABLE IF NOT EXISTS {database}.{table} (
            asset_id                        String,
            asset_name                      String,
            asset_country                   String,
            asset_sector                    String,
            capacity_mw                     Float64,
            latitude                        Float64,
            longitude                       Float64,
            altitude_avg_m                  Float64,
            territorial_type                String,
            economic_level                  String,
            gdp_tier                        Int64,
            solar_irradiance_kwh_m2_yr      Float64,
            wind_speed_avg_ms               Float64,
            regulatory_stability_score      Float64,
            typical_lifespan_years          Float64,
            predicted_lifecycle_stage       String,
            lifecycle_stage_confidence      Float64,
            estimated_retirement_year       Int64,
            estimated_commissioning_year    Int64,
            predicted_remaining_years       Float64,
            predicted_capacity_factor_pct   Float64,
            model_version                   String,
            batch_id                        String
        ) ENGINE = MergeTree()
        ORDER BY asset_id
    """,
}

# Map from DDL table key → (dict_name, key_in_dict).
_TABLE_SOURCES: dict[str, tuple[str, str]] = {
    "ownership_current": ("gold_outputs", "ownership_current"),
    "owner_infrastructure_exposure_snapshot": (
        "gold_outputs",
        "owner_infrastructure_exposure_snapshot",
    ),
    "ml_asset_lifecycle_predictions": (
        "ml_outputs",
        "asset_lifecycle_predictions",
    ),
}

# ---------------------------------------------------------------------------
# ClickHouse type → pandas dtype families that are compatible.
# Used by _validate_dtypes() for early, clear failure on type mismatches.
# ---------------------------------------------------------------------------
_CH_TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "String": ("object", "string"),
    "Float64": ("float64", "float32", "Float64", "Float32"),
    "Int64": ("int64", "int32", "int16", "int8", "Int64", "Int32", "Int16", "Int8"),
    "UInt8": ("uint8", "UInt8", "bool"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_gold_to_clickhouse(
    gold_outputs: dict[str, "pd.DataFrame"],
    ml_outputs: dict[str, "pd.DataFrame"],
) -> None:
    """Load gold analytics tables into ClickHouse using atomic staging-table swap.

    No-ops immediately when ``USE_CLICKHOUSE`` is not ``"true"``.

    A unique ``run_id`` (UUID4 short hex) is generated per invocation and
    appended to staging table names, so concurrent pipeline runs cannot
    interfere with each other's staging tables.

    After every table in this batch has been atomically refreshed, the
    ``run_id`` is appended to ``lakehouse_batch_log``.  Every row in all three
    tables carries a ``batch_id`` column set to the same ``run_id``, so
    downstream OLAP queries can reconstruct a consistent cross-table snapshot
    with::

        WHERE batch_id = (
            SELECT batch_id FROM lakehouse_batch_log
            ORDER BY run_seq DESC LIMIT 1
        )

    ``run_seq`` is derived from ``toUInt64(now64(9))`` evaluated **server-side**
    by ClickHouse at INSERT time, giving a globally consistent nanosecond
    ordering key across all pipeline workers and hosts.

    Parameters
    ----------
    gold_outputs:
        Dict of DataFrames produced by ``build_gold_outputs`` (gold layer).
    ml_outputs:
        Dict of DataFrames produced by ``build_ml_predictions`` (ML layer).

    Raises
    ------
    RuntimeError
        When ``USE_CLICKHOUSE=true`` but the clickhouse-connect package is not
        installed, or the ClickHouse server is unreachable.
    ValueError
        When an expected DataFrame key is missing, when a DataFrame's columns
        do not exactly match the DDL contract, when a column's dtype is
        incompatible with the declared ClickHouse type, or when the configured
        database name is not a safe ClickHouse identifier.
    """
    flag = os.environ.get("USE_CLICKHOUSE", "false").strip().lower()
    if flag != "true":
        return

    cfg = _get_config()
    run_id = uuid.uuid4().hex[:12]
    logger.info(
        "USE_CLICKHOUSE=true — connecting to ClickHouse at %s:%s/%s (run_id=%s, secure=%s)",
        cfg["host"],
        cfg["port"],
        cfg["database"],
        run_id,
        cfg["secure"],
    )

    client = _get_client(cfg)
    client.command(f"CREATE DATABASE IF NOT EXISTS {cfg['database']}")

    all_frames = {"gold_outputs": gold_outputs, "ml_outputs": ml_outputs}

    # Collect ex-live staging table names (old data displaced by EXCHANGE TABLES).
    # These are dropped only after _publish_batch_id succeeds.  If any refresh
    # fails we swap them back to restore the last-known-good state before raising.
    ex_live_tables: list[tuple[str, str]] = []  # [(database, ex_live_staging_name)]
    refreshed_tables: list[str] = []  # live table names successfully exchanged

    try:
        for table_name, (dict_name, key) in _TABLE_SOURCES.items():
            df = all_frames[dict_name].get(key)
            if df is None:
                raise ValueError(
                    f"ClickHouse sink expected key '{key}' in {dict_name}, but it was missing. "
                    "This indicates an upstream pipeline contract change or bug."
                )

            ex_live = _atomic_refresh(
                client, cfg["database"], table_name, df, run_id=run_id
            )
            ex_live_tables.append((cfg["database"], ex_live))
            refreshed_tables.append(table_name)

        # All tables loaded successfully — publish the batch id so downstream
        # queries can obtain a consistent cross-table snapshot marker.
        _publish_batch_id(client, cfg["database"], run_id)

    except Exception:
        # Roll back: re-exchange each successfully-swapped table back to the
        # previous live data so the last-known-good snapshot is restored.
        for live_table, (db, ex_live) in zip(
            refreshed_tables, ex_live_tables[: len(refreshed_tables)]
        ):
            try:
                client.command(f"EXCHANGE TABLES {db}.{live_table} AND {db}.{ex_live}")
                logger.warning(
                    "Rolled back %s.%s to previous live data after partial failure.",
                    db,
                    live_table,
                )
            except Exception as rb_exc:
                logger.error(
                    "Rollback of %s.%s failed: %s — manual recovery may be needed.",
                    db,
                    live_table,
                    rb_exc,
                )
        # Clean up all ex-live staging tables (they now hold the failed run's
        # data after rollback, or the original data if rollback also failed).
        for db, ex_live in ex_live_tables:
            try:
                client.command(f"DROP TABLE IF EXISTS {db}.{ex_live}")
            except Exception:
                pass
        raise

    # Success path: drop the displaced ex-live tables now that the batch is published.
    for db, ex_live in ex_live_tables:
        try:
            client.command(f"DROP TABLE IF EXISTS {db}.{ex_live}")
        except Exception:
            logger.warning("Could not drop ex-live staging table %s.%s; ignoring.", db, ex_live)

    logger.info("ClickHouse sink complete (run_id=%s).", run_id)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Explicit allowed CIDRs for the private-network escape hatch.
# Covers RFC1918 IPv4 ranges and ULA IPv6 (fc00::/7) only.
# Intentionally excludes other ranges that ipaddress.is_private considers
# "private" (e.g. link-local 169.254.x.x, ::1 loopback — loopback is handled
# separately in _get_config) to keep the policy narrow and predictable.
_PRIVATE_CIDRS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("fc00::/7"),  # ULA IPv6
)


def _is_private_network_host(host: str) -> bool:
    """Return True when *host* is safely confined to a private/internal network.

    Accepted as private:
    - The literal string ``"clickhouse"`` (the canonical Docker Compose service
      name used in this project's ``docker-compose.yml``).
    - Any hostname **all** of whose DNS-resolved addresses fall within
      RFC1918 IPv4 (10/8, 172.16/12, 192.168/16) or ULA IPv6 (fc00::/7).

    Resolution uses ``socket.getaddrinfo`` so all returned addresses (IPv4 and
    IPv6, multi-homed) are checked.  A host is considered private only when
    *every* resolved address is private — a host with even one public address
    is rejected.

    Resolution failures are treated conservatively: the host is considered
    public (not private) so the TLS guard still fires.
    """
    if host == "clickhouse":
        return True
    try:
        results = socket.getaddrinfo(host, None)
        if not results:
            return False
        addresses = []
        for _family, _type, _proto, _canonname, sockaddr in results:
            # sockaddr is (address, port) for IPv4 or (address, port, flow, scope) for IPv6.
            addr_str = sockaddr[0]
            try:
                addresses.append(ipaddress.ip_address(addr_str))
            except ValueError:
                return False  # unparseable address → treat as public
        # All resolved addresses must be in one of the allowed private CIDRs.
        return all(
            any(addr in network for network in _PRIVATE_CIDRS)
            for addr in addresses
        )
    except OSError:
        return False


def _get_config() -> dict[str, str | int | bool]:
    """Read ClickHouse connection settings from environment variables.

    TLS policy
    ----------
    ``CLICKHOUSE_SECURE``  — set to ``"true"`` to enable HTTPS/TLS (default:
        ``"false"`` for localhost-only dev; the default port shifts to 8443
        when secure mode is on).
    ``CLICKHOUSE_VERIFY``  — set to ``"false"`` to skip certificate verification
        (only valid when ``CLICKHOUSE_SECURE=true``; useful for self-signed
        certs in dev/staging).  Defaults to ``"true"`` (full verification).

    Non-local connections with ``CLICKHOUSE_SECURE=false`` are **rejected** at
    runtime to prevent accidental credential/data exposure over plaintext HTTP,
    **unless** ``CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK=true`` is also set
    **and** the host resolves to a known-private address or is the canonical
    ``"clickhouse"`` Docker Compose service name.

    The escape hatch is intentionally narrow: it cannot be used to bypass the
    TLS guard for publicly routable or internet-accessible hosts even when the
    flag is set.
    """
    database = os.environ.get("CLICKHOUSE_DATABASE", "lakehouse")
    _validate_identifier(database, env_var="CLICKHOUSE_DATABASE")
    host = os.environ.get("CLICKHOUSE_HOST", "localhost")
    secure_raw = os.environ.get("CLICKHOUSE_SECURE", "false").strip().lower()
    secure = secure_raw == "true"
    verify_raw = os.environ.get("CLICKHOUSE_VERIFY", "true").strip().lower()
    verify = verify_raw != "false"  # default on; only disabled by explicit "false"

    # Default port follows the TLS mode.
    default_port = "8443" if secure else "8123"
    port = int(os.environ.get("CLICKHOUSE_PORT", default_port))

    # Guard: refuse plaintext connections to non-local hosts.
    # "localhost" and loopback addresses are always safe for development.
    # A Docker Compose internal hostname (e.g. "clickhouse") or any host that
    # resolves to a known RFC1918/ULA private address can opt in via
    # CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK=true — but that flag still
    # refuses publicly routable hosts even when set.
    _LOOPBACK = {"localhost", "127.0.0.1", "::1"}
    allow_insecure_private = (
        os.environ.get("CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK", "false")
        .strip()
        .lower()
        == "true"
    )
    if not secure and host not in _LOOPBACK:
        if not (allow_insecure_private and _is_private_network_host(host)):
            raise ValueError(
                f"CLICKHOUSE_SECURE must be 'true' when connecting to a non-local host "
                f"({host!r}). Sending credentials and data over plaintext HTTP to a "
                "remote server is not permitted. Set CLICKHOUSE_SECURE=true (and "
                "optionally CLICKHOUSE_PORT=8443) to enable HTTPS. "
                "For trusted private networks (e.g. Docker Compose), you may set "
                "CLICKHOUSE_ALLOW_INSECURE_PRIVATE_NETWORK=true — but only if the "
                "host resolves to a private/RFC1918 address."
            )

    return {
        "host": host,
        "port": port,
        "database": database,
        "username": os.environ.get("CLICKHOUSE_USER", "default"),
        "password": os.environ.get("CLICKHOUSE_PASSWORD", ""),
        "secure": secure,
        "verify": verify,
    }


def _get_client(cfg: dict):
    """Instantiate and return a clickhouse_connect client.

    Raises RuntimeError with a helpful message when the package is missing or
    the server is unreachable.
    """
    try:
        import clickhouse_connect  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "clickhouse-connect is not installed. "
            "Install with: pip install 'entity-data-lakehouse[clickhouse]'"
        ) from exc

    try:
        client = clickhouse_connect.get_client(
            host=cfg["host"],
            port=cfg["port"],
            username=cfg["username"],
            password=cfg["password"],
            secure=cfg["secure"],
            verify=cfg["verify"],
        )
        return client
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to ClickHouse at {cfg['host']}:{cfg['port']}: {exc}"
        ) from exc


def _atomic_refresh(
    client,
    database: str,
    table_name: str,
    df_insert: "pd.DataFrame",
    *,
    run_id: str,
) -> str:
    """Replace *table_name* atomically using a per-run staging-table swap.

    The staging table name incorporates *run_id* (a UUID4 short hex generated
    once per ``write_gold_to_clickhouse`` call) so that concurrent pipeline
    runs targeting the same database cannot drop or exchange each other's
    staging tables.

    Steps
    -----
    1. Ensure the live table exists (CREATE TABLE IF NOT EXISTS).
    2. Drop any leftover staging table for this run id from a previous aborted
       attempt (should not exist, but defensive).
    3. Create the run-scoped staging table with identical schema.
    4. Insert *df_insert* into the staging table (empty DataFrames are valid —
       they atomically clear the live table on swap).
    5. **Atomically** exchange staging ↔ live with ``EXCHANGE TABLES``.
       Unlike multi-table ``RENAME``, ``EXCHANGE TABLES`` is a single atomic
       operation: readers never see the table missing.
    6. Return the ex-live staging name to the caller without dropping it.
       The caller is responsible for dropping it after all tables succeed
       (or rolling back via a second EXCHANGE TABLES on failure).

    Returns
    -------
    str
        The staging table name that now holds the previous live data
        (``<table_name>__staging_<run_id>``).  The caller must drop it
        after a successful run or exchange it back to roll back on failure.

    Raises
    ------
    RuntimeError
        If ``EXCHANGE TABLES`` fails.  In this case the live table is
        unmodified and the staging table is dropped before raising.
    """
    db = database
    staging = f"{table_name}__staging_{run_id}"

    # Stamp the run-scoped batch_id onto every row before schema validation.
    # This is the mechanism that lets downstream consumers reconstruct a
    # consistent cross-table snapshot: filter WHERE batch_id = <latest batch_id
    # from lakehouse_batch_log>.  The column is declared in the DDL for all
    # three tables so the strict column-set check in _prepare_insert_frame
    # will pass only after this stamp.
    df_stamped = df_insert.copy()
    df_stamped["batch_id"] = run_id

    validated = _prepare_insert_frame(table_name, df_stamped)

    # 1. Ensure live table exists.
    live_ddl = _DDL[table_name].format(database=db, table=table_name)
    client.command(live_ddl)

    # 2. Clean up any leftover staging table from this run (defensive).
    client.command(f"DROP TABLE IF EXISTS {db}.{staging}")

    # 3. Create run-scoped staging table.
    staging_ddl = _DDL[table_name].format(database=db, table=staging)
    client.command(staging_ddl)

    # 4. Insert into staging.
    if not validated.empty:
        client.insert_df(f"{db}.{staging}", validated)
        logger.info("Staged %d rows for %s.%s.", len(validated), db, table_name)
    else:
        logger.info("Staging %s.%s with 0 rows (clears live table).", db, table_name)

    # 5. Atomically exchange staging ↔ live.
    # EXCHANGE TABLES is a true atomic operation (single rename entry in the
    # metadata log); readers see either the old or the new data, never a gap.
    try:
        client.command(f"EXCHANGE TABLES {db}.{table_name} AND {db}.{staging}")
    except Exception as exc:
        # Clean up staging before propagating so it does not litter the schema.
        try:
            client.command(f"DROP TABLE IF EXISTS {db}.{staging}")
        except Exception:
            pass
        raise RuntimeError(
            f"EXCHANGE TABLES failed for {db}.{table_name}: {exc}. "
            "Ensure your ClickHouse version supports EXCHANGE TABLES "
            "(available since 20.5 on MergeTree with allow_experimental_exchange_tables=1 "
            "or natively on 22.6+). The live table is unmodified."
        ) from exc
    logger.debug("Atomically exchanged %s.%s ← staging.", db, table_name)

    # The staging slot now holds the previous live data.  Return its name to
    # the caller — it will be dropped after all tables succeed, or used to
    # roll back if a later table refresh fails.
    logger.info("Refreshed %s.%s (%d rows).", db, table_name, len(validated))
    return staging


def _publish_batch_id(client, database: str, run_id: str) -> None:
    """Record the latest successful batch id in a single-row tracking table.

    Called only after **all** tables for this run have been atomically
    refreshed.  Downstream OLAP queries that need a consistent cross-table
    snapshot can read the current batch id from
    ``<database>.lakehouse_batch_log`` (latest by ``run_seq``).

    ``run_seq`` is ``toUInt64(now64(9))`` — a nanosecond timestamp evaluated
    by the **ClickHouse server** at INSERT time.  Because the value is assigned
    server-side it is globally ordered across all pipeline workers and hosts;
    no client clock skew or NTP adjustment can produce an out-of-order entry.

    This is a lightweight append — not a view swap — so it never disrupts
    readers and does not require EXCHANGE TABLES support.
    """
    db = database
    client.command(
        f"CREATE TABLE IF NOT EXISTS {db}.lakehouse_batch_log "
        f"(batch_id String, loaded_at DateTime64(3) DEFAULT now64(3), "
        f"run_seq UInt64 DEFAULT toUInt64(now64(9))) "
        f"ENGINE = MergeTree() ORDER BY run_seq"
    )
    # Omit run_seq from the column list — ClickHouse evaluates the DEFAULT
    # expression server-side, giving a globally consistent ordering key.
    client.command(
        f"INSERT INTO {db}.lakehouse_batch_log (batch_id) VALUES ('{run_id}')"
    )
    logger.info("Published batch id %s to %s.lakehouse_batch_log.", run_id, db)


def _validate_identifier(identifier: str, *, env_var: str) -> None:
    """Raise ValueError if *identifier* is not a safe ClickHouse identifier."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(
            f"Invalid ClickHouse identifier in {env_var}: {identifier!r}. "
            "Only letters, numbers, and underscores are allowed, and the first "
            "character must be a letter or underscore."
        )


def _prepare_insert_frame(table_name: str, df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate *df* schema and dtypes, then return a copy in DDL column order.

    Checks performed
    ----------------
    - Exact column-set match (missing *and* extra columns are rejected).
    - Each column's pandas dtype must be compatible with the declared
      ClickHouse type (e.g. an Int64 column must not arrive as ``object``).

    This avoids silent schema drift and ensures ClickHouse receives the same
    logical contract as DuckDB.
    """
    ddl_cols = _parse_ddl_columns(table_name)
    expected_cols = list(ddl_cols.keys())
    expected = set(expected_cols)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"ClickHouse sink schema mismatch for table '{table_name}'. "
            f"missing={missing} extra={extra}. "
            "Update the DDL in clickhouse_sink.py to match the gold output contract, "
            "or fix the upstream pipeline."
        )

    _validate_dtypes(table_name, df, ddl_cols)

    return df.loc[:, expected_cols].copy()


def _validate_dtypes(
    table_name: str,
    df: "pd.DataFrame",
    ddl_cols: dict[str, str],
) -> None:
    """Raise ValueError for columns whose pandas dtype is incompatible with the DDL type.

    Only columns present in both *df* and *ddl_cols* are checked; callers
    should call this after column-presence validation.

    Known type families are validated; unknown ClickHouse types are skipped
    rather than falsely failing (forward-compatibility).
    """
    errors: list[str] = []
    for col, ch_type in ddl_cols.items():
        if col not in df.columns:
            continue
        allowed = _CH_TYPE_FAMILIES.get(ch_type)
        if allowed is None:
            # Unknown type — skip rather than block.
            continue
        actual_dtype = str(df[col].dtype)
        if actual_dtype not in allowed:
            errors.append(
                f"  column '{col}': declared {ch_type}, "
                f"pandas dtype is '{actual_dtype}' (allowed: {allowed})"
            )
    if errors:
        raise ValueError(
            f"ClickHouse sink dtype mismatch for table '{table_name}':\n"
            + "\n".join(errors)
            + "\nConvert columns upstream or update the DDL."
        )


def _parse_ddl_columns(table_name: str) -> dict[str, str]:
    """Extract (column_name → ClickHouse type) from the DDL template.

    Only handles simple ``name  Type`` lines between the CREATE TABLE line
    (which ends with ``(``) and the closing ``)``.  Sufficient for the narrow
    schemas defined above.
    """
    ddl = _DDL[table_name]
    cols: dict[str, str] = {}
    in_cols = False
    for line in ddl.splitlines():
        stripped = line.strip()
        # The CREATE TABLE line ends with '(' — that marks the start of columns.
        if stripped.endswith("(") and "CREATE" in stripped.upper():
            in_cols = True
            continue
        if stripped.startswith(")") or stripped.upper().startswith("ENGINE"):
            in_cols = False
            continue
        if in_cols and stripped and not stripped.startswith("--"):
            # e.g.  "entity_id       String,"
            parts = stripped.rstrip(",").split()
            if len(parts) >= 2:
                cols[parts[0]] = parts[1]
    return cols
