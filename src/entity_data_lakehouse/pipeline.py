from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from pathlib import Path

import duckdb

from .bronze import ingest_sample_data
from .gold import build_gold_outputs
from .ml import build_ml_predictions
from .public_safety import scan_public_safety
from .silver import build_silver_outputs

logger = logging.getLogger(__name__)

_VALID_PUBLISH_MODES = {"dry_run", "commit"}

# Tables that the ClickHouse sink will attempt to refresh (in order).
_SINK_TABLES = [
    "ownership_current",
    "owner_infrastructure_exposure_snapshot",
    "ml_asset_lifecycle_predictions",
]


def run_pipeline(
    repo_root: Path,
    *,
    publish_mode: str = "commit",
    report_path: Path | None = None,
) -> dict[str, int]:
    """Run the full bronze → silver → gold → ML pipeline.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.  All data paths are resolved
        relative to this directory.
    publish_mode:
        ``"commit"`` (default) — full pipeline with all disk writes and
        optional ClickHouse sink.  Behaviour is identical to the pre-publish-
        mode baseline.

        ``"dry_run"`` — all computation and contract validation runs but
        **no disk writes** are performed (no parquet files, no DuckDB tables,
        no ClickHouse mutations).  The only artifact produced is
        ``publish_report.json``.  Useful for CI preflight, demo review, and
        manual pre-publish checks.
    report_path:
        Where to write ``publish_report.json``.  Defaults to
        ``{repo_root}/gold/publish_report.json``.  The parent directory is
        created if it does not exist.

    Returns
    -------
    dict[str, int]
        Row counts keyed by layer:
        ``entity_master_rows``, ``asset_master_rows``,
        ``relationship_edge_rows``, ``gold_rows``, ``ml_prediction_rows``.

    Raises
    ------
    ValueError
        On invalid ``publish_mode``, failed contract validations, or a failed
        public-safety scan.
    """
    if publish_mode not in _VALID_PUBLISH_MODES:
        raise ValueError(
            f"Invalid publish_mode {publish_mode!r}. "
            f"Must be one of: {sorted(_VALID_PUBLISH_MODES)}"
        )

    dry_run = publish_mode == "dry_run"
    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    if report_path is None:
        report_path = repo_root / "gold" / "publish_report.json"

    clickhouse_enabled = os.environ.get("USE_CLICKHOUSE", "false").strip().lower() == "true"

    # Initialise the report skeleton — fields are filled in as the pipeline
    # progresses.  Written at the end regardless of success or failure.
    report: dict = {
        "schema_version": "1",
        "report_timestamp": started_at,
        "run_id": run_id,
        "publish_mode": publish_mode,
        "status": "failed",  # overwritten to "success" at end
        "tables_attempted": _SINK_TABLES,
        "row_counts": {},
        "rollback_status": "not_applicable",
        "sink_target": {
            "clickhouse_enabled": clickhouse_enabled,
            "tables_refreshed": [],
            "batch_id": None,
            "status": "skipped" if not clickhouse_enabled else (
                "dry_run_validated" if dry_run else "not_started"
            ),
            "schema_validations": [],
        },
        "public_safety": {"status": "pending", "findings": []},
        "artifacts_written": [],
    }

    contracts_root = repo_root / "contracts"
    sample_root = repo_root / "sample_data"
    reference_root = repo_root / "reference_data"

    try:
        # ------------------------------------------------------------------
        # Bronze
        # ------------------------------------------------------------------
        bronze_parquet_root = repo_root / "bronze"
        ingest_sample_data(
            sample_root=sample_root,
            bronze_root=bronze_parquet_root,
            contract_path=contracts_root / "bronze_source_record.schema.json",
            dry_run=dry_run,
        )

        # ------------------------------------------------------------------
        # Silver
        # ------------------------------------------------------------------
        silver_outputs = build_silver_outputs(
            sample_root=sample_root,
            silver_root=repo_root / "silver",
            contract_paths={
                "entity_observations": contracts_root / "entity_observations.schema.json",
                "entity_master": contracts_root / "entity_master.schema.json",
                "asset_master": contracts_root / "asset_master.schema.json",
                "ownership_observations": contracts_root
                / "ownership_observations.schema.json",
                "relationship_edges": contracts_root / "relationship_edges.schema.json",
            },
            dry_run=dry_run,
        )

        # ------------------------------------------------------------------
        # Gold
        # ------------------------------------------------------------------
        gold_outputs, gold_artifacts = build_gold_outputs(
            gold_root=repo_root / "gold",
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
                "ownership_lifecycle": contracts_root / "ownership_lifecycle.schema.json",
                "ownership_history_scd2": contracts_root
                / "ownership_history_scd2.schema.json",
                "ownership_current": contracts_root / "ownership_current.schema.json",
                "owner_infrastructure_exposure_snapshot": contracts_root
                / "owner_infrastructure_exposure_snapshot.schema.json",
            },
            dry_run=dry_run,
        )

        # gold_artifacts is collected incrementally inside build_gold_outputs().
        # On success this reflects every file written.  If build_gold_outputs()
        # raises mid-write it attaches a partial list to __gold_artifacts__ on the
        # exception; the except-block below picks that up instead, so failure
        # reports are always accurate regardless of where gold writing stops.
        report["artifacts_written"] = list(gold_artifacts)

        # ------------------------------------------------------------------
        # Safety gate — runs before ML so telemetry never fires on failure
        # ------------------------------------------------------------------
        safety_findings = scan_public_safety(repo_root)
        if safety_findings:
            report["public_safety"] = {"status": "failed", "findings": safety_findings}
            raise ValueError("Public-safety scan failed:\n" + "\n".join(safety_findings))
        report["public_safety"] = {"status": "passed", "findings": []}

        # ------------------------------------------------------------------
        # ML
        # ------------------------------------------------------------------
        ml_outputs = build_ml_predictions(
            gold_root=repo_root / "gold",
            silver_outputs=silver_outputs,
            gold_outputs=gold_outputs,
            reference_root=reference_root,
            contract_paths={
                "asset_lifecycle_predictions": contracts_root
                / "asset_lifecycle_predictions.schema.json",
            },
            dry_run=dry_run,
        )

        ml_predictions = ml_outputs["asset_lifecycle_predictions"]

        # ML parquet is written by build_ml_predictions(); record it now.
        if not dry_run:
            report["artifacts_written"].append("gold/dw/asset_lifecycle_predictions.parquet")

        # ------------------------------------------------------------------
        # Row counts (computed from in-memory frames — always available)
        # ------------------------------------------------------------------
        row_counts = {
            "entity_master_rows": len(silver_outputs["entity_master"]),
            "asset_master_rows": len(silver_outputs["asset_master"]),
            "relationship_edge_rows": len(silver_outputs["relationship_edges"]),
            "gold_rows": len(gold_outputs["owner_infrastructure_exposure_snapshot"]),
            "ml_prediction_rows": len(ml_predictions),
        }
        report["row_counts"] = row_counts

        # ------------------------------------------------------------------
        # DuckDB registration (commit only)
        # ------------------------------------------------------------------
        if not dry_run:
            duckdb_path = repo_root / "gold" / "entity_lakehouse.duckdb"
            con = duckdb.connect(str(duckdb_path))
            try:
                con.execute(
                    "CREATE OR REPLACE TABLE ml_asset_lifecycle_predictions "
                    "AS SELECT * FROM ml_predictions"
                )
            finally:
                con.close()
            logger.info(
                "Registered ML predictions in DuckDB: %d rows.", len(ml_predictions)
            )

        # ------------------------------------------------------------------
        # ClickHouse sink
        # ------------------------------------------------------------------
        from .clickhouse_sink import validate_sink_schema, write_gold_to_clickhouse

        if dry_run:
            # Validate schemas only — no connection, no mutation.
            schema_validations = validate_sink_schema(gold_outputs, ml_outputs)
            all_passed = all(v["status"] == "passed" for v in schema_validations)
            report["sink_target"]["schema_validations"] = schema_validations
            report["sink_target"]["status"] = (
                "dry_run_validated" if all_passed else "dry_run_schema_failed"
            )
            logger.info(
                "dry_run: ClickHouse schema validation %s.",
                "passed" if all_passed else "FAILED",
            )
            if not all_passed:
                failed = [v["table"] for v in schema_validations if v["status"] != "passed"]
                raise ValueError(
                    f"dry_run: ClickHouse schema validation failed for tables: {failed}"
                )
        else:
            # Full atomic refresh — only runs when USE_CLICKHOUSE=true.
            sink_summary = write_gold_to_clickhouse(gold_outputs, ml_outputs)
            report["sink_target"]["tables_refreshed"] = sink_summary["tables_refreshed"]
            report["sink_target"]["batch_id"] = sink_summary["batch_id"]
            report["sink_target"]["status"] = sink_summary["status"]
            report["rollback_status"] = sink_summary["rollback_status"]

        report["status"] = "success"
        logger.info(
            "Pipeline complete (publish_mode=%s, run_id=%s).", publish_mode, run_id
        )

    except Exception as exc:
        report["status"] = "failed"
        logger.error(
            "Pipeline failed (publish_mode=%s, run_id=%s): %s",
            publish_mode,
            run_id,
            exc,
        )
        # If build_gold_outputs() failed mid-write it attaches whatever was
        # already written to __gold_artifacts__.  Use that to overwrite the
        # (possibly empty) report list so recovery/cleanup info is accurate.
        gold_partial = getattr(exc, "__gold_artifacts__", None)
        if gold_partial is not None:
            report["artifacts_written"] = list(gold_partial)
        # If the ClickHouse sink attached structured rollback metadata, inject
        # it into the failure report before writing.
        sink_summary = getattr(exc, "__sink_summary__", None)
        if sink_summary is not None:
            report["sink_target"]["tables_refreshed"] = sink_summary.get("tables_refreshed", [])
            report["sink_target"]["batch_id"] = sink_summary.get("batch_id")
            report["sink_target"]["status"] = sink_summary.get("status", "failed")
            report["rollback_status"] = sink_summary.get("rollback_status", "not_applicable")
        elif report["sink_target"]["status"] == "not_started":
            # Failure occurred before the sink was reached or before __sink_summary__
            # was attached (e.g. ClickHouse config/connection error).  Overwrite the
            # non-terminal initialisation value with a terminal "failed" so the report
            # is always machine-readable without ambiguity.
            report["sink_target"]["status"] = "failed"
        _write_report(report, report_path, dry_run=dry_run)
        raise

    _write_report(report, report_path, dry_run=dry_run)
    return row_counts


def _write_report(report: dict, report_path: Path, *, dry_run: bool = False) -> None:
    """Write publish_report.json to *report_path*, creating parent dirs as needed.

    In ``dry_run`` mode the report is the *only* permitted artifact, so a write
    failure is fatal and raises ``RuntimeError``.  In ``commit`` mode a write
    failure is non-fatal (all real artifacts are already on disk), so only a
    warning is logged.
    """
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Publish report written to %s", report_path)
    except Exception as exc:
        msg = f"Could not write publish report to {report_path}: {exc}"
        if dry_run:
            raise RuntimeError(msg) from exc
        logger.warning(msg)
