from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from .bronze import ingest_sample_data
from .gold import build_gold_outputs
from .ml import build_ml_predictions
from .public_safety import scan_public_safety
from .silver import build_silver_outputs

logger = logging.getLogger(__name__)


def run_pipeline(repo_root: Path) -> dict[str, int]:
    contracts_root = repo_root / "contracts"
    sample_root = repo_root / "sample_data"
    reference_root = repo_root / "reference_data"

    ingest_sample_data(
        sample_root=sample_root,
        bronze_root=repo_root / "bronze",
        contract_path=contracts_root / "bronze_source_record.schema.json",
    )

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
    )

    gold_outputs = build_gold_outputs(
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
    )

    # Safety gate: run BEFORE build_ml_predictions so that even Langfuse/off-box
    # telemetry emitted during inference never fires when the scan would fail.
    # Parquet/DuckDB files written so far are local-only and are not sent anywhere.
    safety_findings = scan_public_safety(repo_root)
    if safety_findings:
        raise ValueError("Public-safety scan failed:\n" + "\n".join(safety_findings))

    # ML extrapolation: train on synthetic reference data, predict for real assets.
    ml_outputs = build_ml_predictions(
        gold_root=repo_root / "gold",
        silver_outputs=silver_outputs,
        gold_outputs=gold_outputs,
        reference_root=reference_root,
        contract_paths={
            "asset_lifecycle_predictions": contracts_root
            / "asset_lifecycle_predictions.schema.json",
        },
    )

    # Register ML predictions in DuckDB alongside the gold warehouse tables.
    duckdb_path = repo_root / "gold" / "entity_lakehouse.duckdb"
    ml_predictions = ml_outputs["asset_lifecycle_predictions"]
    con = duckdb.connect(str(duckdb_path))
    con.execute(
        "CREATE OR REPLACE TABLE ml_asset_lifecycle_predictions AS SELECT * FROM ml_predictions"
    )
    con.close()
    logger.info("Registered ML predictions in DuckDB: %d rows.", len(ml_predictions))

    # Optional ClickHouse sink — no-ops unless USE_CLICKHOUSE=true.
    from .clickhouse_sink import write_gold_to_clickhouse

    write_gold_to_clickhouse(gold_outputs, ml_outputs)

    return {
        "entity_master_rows": len(silver_outputs["entity_master"]),
        "asset_master_rows": len(silver_outputs["asset_master"]),
        "relationship_edge_rows": len(silver_outputs["relationship_edges"]),
        "gold_rows": len(gold_outputs["owner_infrastructure_exposure_snapshot"]),
        "ml_prediction_rows": len(ml_predictions),
    }
