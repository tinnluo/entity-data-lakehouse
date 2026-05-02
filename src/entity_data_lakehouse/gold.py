from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .contracts import validate_dataframe
from .utils import stable_id


ENTITY_SOURCE_PRIORITY = {
    "registry_entities": 1,
    "entity_hierarchy": 2,
    "infrastructure_assets": 3,
}


def _snapshot_metadata(snapshot_dates: list[str]) -> dict[str, tuple[str, bool]]:
    latest = max(snapshot_dates)
    return {
        snapshot_date: (f"SNAPSHOT_{index:03d}", snapshot_date == latest)
        for index, snapshot_date in enumerate(sorted(snapshot_dates), start=1)
    }


def _entity_scd4(entity_observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    snapshots = sorted(entity_observations["snapshot_date"].unique())
    metadata = _snapshot_metadata(snapshots)
    comprehensive_rows: list[dict] = []
    prior_active: dict[tuple[str, str], dict] = {}

    for sequence_number, snapshot_date in enumerate(snapshots, start=1):
        snapshot_df = entity_observations[entity_observations["snapshot_date"] == snapshot_date].copy()
        current_rows = snapshot_df.to_dict(orient="records")
        current_by_key = {(row["entity_id"], row["observation_source"]): row for row in current_rows}
        snapshot_version, is_latest = metadata[snapshot_date]

        for key, row in current_by_key.items():
            prior = prior_active.get(key)
            if prior is None:
                status = "NEW"
            elif row["observation_key_hash"] == prior["observation_key_hash"] and row["entity_name"] == prior["entity_name"]:
                status = "UNCHANGED"
            else:
                status = "CHANGED"
            comprehensive_rows.append(
                {
                    **row,
                    "snapshot_version": snapshot_version,
                    "snapshot_sequence_number": sequence_number,
                    "is_latest_snapshot": is_latest,
                    "change_status_backward": status,
                    "is_dropped": False,
                }
            )

        dropped_keys = set(prior_active) - set(current_by_key)
        for key in sorted(dropped_keys):
            prior = prior_active[key]
            comprehensive_rows.append(
                {
                    **prior,
                    "snapshot_date": snapshot_date,
                    "source_version": snapshot_date,
                    "snapshot_version": snapshot_version,
                    "snapshot_sequence_number": sequence_number,
                    "is_latest_snapshot": is_latest,
                    "change_status_backward": "DROPPED",
                    "is_dropped": True,
                }
            )

        prior_active = current_by_key

    comprehensive = pd.DataFrame(comprehensive_rows).sort_values(
        ["snapshot_sequence_number", "entity_id", "observation_source", "source_record_id", "is_dropped"]
    ).reset_index(drop=True)
    comprehensive["is_latest_snapshot"] = comprehensive["is_latest_snapshot"].astype("bool")
    comprehensive["is_dropped"] = comprehensive["is_dropped"].astype("bool")

    latest_snapshot = max(snapshots)
    latest_rows = comprehensive[
        (comprehensive["snapshot_date"] == latest_snapshot) & (~comprehensive["is_dropped"])
    ].copy()
    latest_rows["source_priority"] = latest_rows["observation_source"].map(ENTITY_SOURCE_PRIORITY).fillna(99)
    current_master = (
        latest_rows.sort_values(["entity_id", "source_priority", "entity_name"])
        .groupby("entity_id", as_index=False)
        .first()
        .loc[
            :,
            [
                "entity_id",
                "entity_name",
                "normalized_name",
                "country_code",
                "entity_type",
                "registry_entity_id",
                "lei",
                "source_entity_id",
                "observation_source",
                "snapshot_date",
                "entity_resolution_method",
            ],
        ]
        .rename(
            columns={
                "observation_source": "current_observation_source",
                "snapshot_date": "current_snapshot_date",
            }
        )
        .sort_values(["entity_name", "entity_id"])
        .reset_index(drop=True)
    )

    event_rows: list[dict] = []
    for snapshot_date in snapshots:
        snapshot_df = comprehensive[comprehensive["snapshot_date"] == snapshot_date]
        for entity_id, group in snapshot_df.groupby("entity_id"):
            if (~group["is_dropped"]).any():
                active = group[~group["is_dropped"]]
                if (active["change_status_backward"] == "CHANGED").any():
                    event_type = "CHANGED"
                elif (active["change_status_backward"] == "NEW").any():
                    event_type = "NEW"
                else:
                    event_type = "UNCHANGED"
            else:
                event_type = "DROPPED"
            event_rows.append(
                {
                    "entity_id": entity_id,
                    "snapshot_date": snapshot_date,
                    "snapshot_sequence_number": int(group["snapshot_sequence_number"].iloc[0]),
                    "event_type": event_type,
                }
            )
    event_log = pd.DataFrame(event_rows).sort_values(
        ["snapshot_sequence_number", "entity_id"]
    ).reset_index(drop=True)
    event_log["snapshot_sequence_number"] = event_log["snapshot_sequence_number"].astype("int64")
    return comprehensive, current_master, event_log


def _attach_forward_status(comprehensive: pd.DataFrame) -> pd.DataFrame:
    rows = comprehensive.copy()
    rows["change_status_forward"] = "CURRENT"
    for observation_key, group in rows.groupby("observation_key", sort=False):
        indices = list(group.sort_values("snapshot_sequence_number").index)
        for position, index in enumerate(indices):
            current = rows.loc[index]
            if position == len(indices) - 1:
                rows.loc[index, "change_status_forward"] = "ABSENCE" if current["is_dropped"] else "CURRENT"
                continue
            nxt = rows.loc[indices[position + 1]]
            if nxt["is_dropped"]:
                rows.loc[index, "change_status_forward"] = "DROPPED"
            elif current["row_hash"] != nxt["row_hash"]:
                rows.loc[index, "change_status_forward"] = "CHANGED"
            else:
                rows.loc[index, "change_status_forward"] = "CURRENT"
    return rows


def _ownership_scd4(ownership_observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = sorted(ownership_observations["snapshot_date"].unique())
    metadata = _snapshot_metadata(snapshots)
    comprehensive_rows: list[dict] = []
    prior_active: dict[str, dict] = {}

    for sequence_number, snapshot_date in enumerate(snapshots, start=1):
        snapshot_df = ownership_observations[ownership_observations["snapshot_date"] == snapshot_date].copy()
        current_rows = snapshot_df.to_dict(orient="records")
        current_by_key = {row["observation_key"]: row for row in current_rows}
        snapshot_version, is_latest = metadata[snapshot_date]

        for observation_key, row in current_by_key.items():
            prior = prior_active.get(observation_key)
            if prior is None:
                status = "NEW"
            elif row["row_hash"] == prior["row_hash"]:
                status = "UNCHANGED"
            else:
                status = "CHANGED"
            comprehensive_rows.append(
                {
                    **row,
                    "snapshot_version": snapshot_version,
                    "snapshot_sequence_number": sequence_number,
                    "is_latest_snapshot": is_latest,
                    "change_status_backward": status,
                    "row_change_status_backward": status,
                    "is_dropped": False,
                }
            )

        dropped = set(prior_active) - set(current_by_key)
        for observation_key in sorted(dropped):
            prior = prior_active[observation_key]
            comprehensive_rows.append(
                {
                    **prior,
                    "snapshot_date": snapshot_date,
                    "observation_date": snapshot_date,
                    "source_version": snapshot_date,
                    "snapshot_version": snapshot_version,
                    "snapshot_sequence_number": sequence_number,
                    "is_latest_snapshot": is_latest,
                    "change_status_backward": "DROPPED",
                    "row_change_status_backward": "DROPPED",
                    "is_dropped": True,
                }
            )

        prior_active = current_by_key

    comprehensive = pd.DataFrame(comprehensive_rows).sort_values(
        ["snapshot_sequence_number", "owner_entity_id", "asset_id", "is_dropped"]
    ).reset_index(drop=True)
    comprehensive["is_latest_snapshot"] = comprehensive["is_latest_snapshot"].astype("bool")
    comprehensive["is_dropped"] = comprehensive["is_dropped"].astype("bool")
    comprehensive = _attach_forward_status(comprehensive)
    comprehensive["row_change_status_forward"] = comprehensive["change_status_forward"]

    sequence_map = {snapshot: index for index, snapshot in enumerate(snapshots, start=1)}
    lifecycle_rows: list[dict] = []
    for lifecycle_key, group in comprehensive.groupby("lifecycle_key", sort=False):
        active = group[~group["is_dropped"]].copy()
        appearance_snapshots = sorted(active["snapshot_date"].unique())
        first_snapshot = appearance_snapshots[0]
        last_snapshot = appearance_snapshots[-1]
        first_seq = sequence_map[first_snapshot]
        last_seq = sequence_map[last_snapshot]
        total_available = last_seq - first_seq + 1
        total_appearances = len(appearance_snapshots)
        gap_periods = total_available - total_appearances
        flags = [snap in set(appearance_snapshots) for snap in snapshots[first_seq - 1 : last_seq]]
        max_consecutive = 0
        current_run = 0
        for flag in flags:
            if flag:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 0
        present_latest = appearance_snapshots[-1] == snapshots[-1]
        consecutive_current = current_run if present_latest else 0
        presence_rate = total_appearances / total_available
        recency_weight = 1.0 if present_latest else 0.8
        gap_penalty = max(0.0, 1.0 - (gap_periods * 0.1))
        reliability_score = round(presence_rate * recency_weight * gap_penalty, 3)
        if present_latest and total_appearances == 1:
            lifecycle_status = "NEW"
        elif present_latest and gap_periods == 0:
            lifecycle_status = "ACTIVE"
        elif present_latest:
            lifecycle_status = "INTERMITTENT"
        else:
            lifecycle_status = "DROPPED"
        latest_active = active.sort_values("snapshot_sequence_number").iloc[-1]
        lifecycle_rows.append(
            {
                "lifecycle_key": lifecycle_key,
                "owner_entity_id": latest_active["owner_entity_id"],
                "asset_id": latest_active["asset_id"],
                "observation_sources": ",".join(sorted(group["observation_source"].unique())),
                "source_count": int(group["observation_source"].nunique()),
                "first_appearance_version": metadata[first_snapshot][0],
                "first_appearance_sequence": int(first_seq),
                "last_appearance_version": metadata[last_snapshot][0],
                "last_appearance_sequence": int(last_seq),
                "total_appearances": int(total_appearances),
                "total_snapshots_available": int(total_available),
                "presence_rate": round(presence_rate, 3),
                "consecutive_appearances_current": int(consecutive_current),
                "max_consecutive_appearances": int(max_consecutive),
                "gap_periods": int(gap_periods),
                "reliability_score": reliability_score,
                "lifecycle_status": lifecycle_status,
            }
        )
    lifecycle = pd.DataFrame(lifecycle_rows).sort_values(["owner_entity_id", "asset_id"]).reset_index(drop=True)
    return comprehensive, lifecycle


def _ownership_scd2(ownership_comprehensive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_rows = ownership_comprehensive[~ownership_comprehensive["is_dropped"]].copy()
    snapshots = sorted(active_rows["snapshot_date"].unique())
    by_snapshot = {snapshot: index for index, snapshot in enumerate(snapshots, start=1)}
    history_rows: list[dict] = []

    for business_key_hash, group in active_rows.groupby("observation_key_hash", sort=False):
        ordered = (
            group.sort_values("snapshot_sequence_number")
            .drop_duplicates(subset=["snapshot_date"], keep="last")
            .reset_index(drop=True)
        )
        current_version = 0
        current_record: dict | None = None
        previous_snapshot = ""
        for row in ordered.to_dict(orient="records"):
            change_payload = (
                row["owner_entity_id"],
                row["asset_id"],
                row["asset_name"],
                row["asset_country"],
                row["asset_sector"],
                round(float(row["ownership_pct"]), 2),
            )
            current_snapshot = row["snapshot_date"]
            if current_record is None:
                current_version = 1
                current_record = {
                    "ownership_sk": stable_id("osk", business_key_hash, current_version),
                    "business_key_hash": business_key_hash,
                    "owner_entity_id": row["owner_entity_id"],
                    "owner_entity_name": row["owner_entity_name"],
                    "asset_id": row["asset_id"],
                    "asset_name": row["asset_name"],
                    "asset_country": row["asset_country"],
                    "asset_sector": row["asset_sector"],
                    "capacity_mw": float(row["capacity_mw"]),
                    "ownership_pct": round(float(row["ownership_pct"]), 2),
                    "observation_source": row["observation_source"],
                    "effective_date": current_snapshot,
                    "expiry_date": "",
                    "is_current_flag": "Y",
                    "version_number": current_version,
                    "change_reason": "NEW",
                    "dw_batch_id": row["snapshot_version"],
                }
                previous_snapshot = current_snapshot
                continue

            snapshot_gap = by_snapshot[current_snapshot] - by_snapshot[previous_snapshot]
            current_payload = (
                current_record["owner_entity_id"],
                current_record["asset_id"],
                current_record["asset_name"],
                current_record["asset_country"],
                current_record["asset_sector"],
                current_record["ownership_pct"],
            )
            if snapshot_gap > 1 or change_payload != current_payload:
                current_record["expiry_date"] = current_snapshot
                current_record["is_current_flag"] = "N"
                history_rows.append(current_record)
                current_version += 1
                current_record = {
                    "ownership_sk": stable_id("osk", business_key_hash, current_version),
                    "business_key_hash": business_key_hash,
                    "owner_entity_id": row["owner_entity_id"],
                    "owner_entity_name": row["owner_entity_name"],
                    "asset_id": row["asset_id"],
                    "asset_name": row["asset_name"],
                    "asset_country": row["asset_country"],
                    "asset_sector": row["asset_sector"],
                    "capacity_mw": float(row["capacity_mw"]),
                    "ownership_pct": round(float(row["ownership_pct"]), 2),
                    "observation_source": row["observation_source"],
                    "effective_date": current_snapshot,
                    "expiry_date": "",
                    "is_current_flag": "Y",
                    "version_number": current_version,
                    "change_reason": "REOPENED" if snapshot_gap > 1 else "CHANGED",
                    "dw_batch_id": row["snapshot_version"],
                }
            previous_snapshot = current_snapshot

        if current_record is not None:
            missing_after_last = set(snapshots[by_snapshot[previous_snapshot] :]) - set(ordered["snapshot_date"])
            if missing_after_last:
                current_record["expiry_date"] = min(missing_after_last)
                current_record["is_current_flag"] = "N"
            history_rows.append(current_record)

    history = pd.DataFrame(history_rows).sort_values(
        ["owner_entity_id", "asset_id", "version_number"]
    ).reset_index(drop=True)
    history["version_number"] = history["version_number"].astype("int64")
    current = history[history["is_current_flag"] == "Y"].copy().reset_index(drop=True)
    return history, current


def _derive_owner_mart(
    ownership_history: pd.DataFrame,
    snapshot_dates: list[str],
    contract_path: Path,
) -> pd.DataFrame:
    all_rows: list[dict] = []
    prior_by_grain: dict[tuple[str, str, str], dict] = {}

    for snapshot_date in sorted(snapshot_dates):
        active = ownership_history[
            (ownership_history["effective_date"] <= snapshot_date)
            & (
                (ownership_history["expiry_date"] == "")
                | (ownership_history["expiry_date"] > snapshot_date)
            )
        ].copy()
        if active.empty:
            current_by_grain = {}
        else:
            active["owned_capacity_component"] = active["capacity_mw"] * active["ownership_pct"] / 100.0
            active["controlled_component"] = (active["ownership_pct"] >= 50).astype(int)
            aggregated = (
                active.groupby(["owner_entity_id", "asset_country", "asset_sector"], as_index=False)
                .agg(
                    asset_count=("asset_id", "nunique"),
                    controlled_asset_count=("controlled_component", "sum"),
                    owned_capacity_mw=("owned_capacity_component", "sum"),
                    average_ownership_pct=("ownership_pct", "mean"),
                    relationship_count=("ownership_sk", "count"),
                )
                .sort_values(["owner_entity_id", "asset_country", "asset_sector"])
                .reset_index(drop=True)
            )
            current_by_grain = {
                (row["owner_entity_id"], row["asset_country"], row["asset_sector"]): row
                for row in aggregated.to_dict(orient="records")
            }

        for grain, row in current_by_grain.items():
            prior = prior_by_grain.get(grain)
            metric_keys = [
                "asset_count",
                "controlled_asset_count",
                "owned_capacity_mw",
                "average_ownership_pct",
                "relationship_count",
            ]
            if prior is None:
                status = "NEW"
            elif all(row[key] == prior[key] for key in metric_keys):
                status = "UNCHANGED"
            else:
                status = "CHANGED"
            all_rows.append({**row, "snapshot_date": snapshot_date, "change_status_vs_prior_snapshot": status})

        if prior_by_grain:
            dropped = set(prior_by_grain) - set(current_by_grain)
            for grain in sorted(dropped):
                prior = prior_by_grain[grain]
                all_rows.append(
                    {
                        "snapshot_date": snapshot_date,
                        "owner_entity_id": prior["owner_entity_id"],
                        "asset_country": prior["asset_country"],
                        "asset_sector": prior["asset_sector"],
                        "asset_count": 0,
                        "controlled_asset_count": 0,
                        "owned_capacity_mw": 0.0,
                        "average_ownership_pct": 0.0,
                        "relationship_count": 0,
                        "change_status_vs_prior_snapshot": "DROPPED",
                    }
                )

        prior_by_grain = current_by_grain

    mart = pd.DataFrame(all_rows).sort_values(
        ["snapshot_date", "owner_entity_id", "asset_country", "asset_sector"]
    ).reset_index(drop=True)
    mart["asset_count"] = mart["asset_count"].astype("int64")
    mart["controlled_asset_count"] = mart["controlled_asset_count"].astype("int64")
    mart["relationship_count"] = mart["relationship_count"].astype("int64")
    validate_dataframe(mart, contract_path)
    return mart


def build_gold_outputs(
    gold_root: Path,
    silver_outputs: dict[str, pd.DataFrame],
    contract_paths: dict[str, Path],
    *,
    dry_run: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Build and validate all gold outputs.

    Returns
    -------
    outputs : dict[str, pd.DataFrame]
        In-memory gold frames (always populated, even in dry_run).
    artifacts_written : list[str]
        Repo-relative paths of every file actually written to disk during this
        call, collected incrementally so that a mid-function failure leaves the
        list accurate up to the point of failure.  Empty in dry_run mode.
    """
    artifacts_written: list[str] = []

    if not dry_run:
        gold_root.mkdir(parents=True, exist_ok=True)
        dw_root = gold_root / "dw"
        dw_root.mkdir(parents=True, exist_ok=True)
    else:
        dw_root = gold_root / "dw"

    entity_comprehensive, entity_current, entity_event_log = _entity_scd4(
        silver_outputs["entity_observations"]
    )
    ownership_comprehensive, ownership_lifecycle = _ownership_scd4(
        silver_outputs["ownership_observations"]
    )
    ownership_history, ownership_current = _ownership_scd2(ownership_comprehensive)
    owner_mart = _derive_owner_mart(
        ownership_history,
        sorted(silver_outputs["ownership_observations"]["snapshot_date"].unique()),
        contract_paths["owner_infrastructure_exposure_snapshot"],
    )

    outputs = {
        "entity_master_comprehensive_scd4": entity_comprehensive,
        "entity_master_current": entity_current,
        "entity_master_event_log": entity_event_log,
        "ownership_comprehensive_scd4": ownership_comprehensive,
        "ownership_lifecycle": ownership_lifecycle,
        "ownership_history_scd2": ownership_history,
        "ownership_current": ownership_current,
    }

    try:
        for name, frame in outputs.items():
            validate_dataframe(frame, contract_paths[name])
            if not dry_run:
                frame.to_parquet(dw_root / f"{name}.parquet", index=False)
                artifacts_written.append(f"gold/dw/{name}.parquet")

        if not dry_run:
            owner_mart.to_parquet(
                gold_root / "owner_infrastructure_exposure_snapshot.parquet", index=False
            )
            artifacts_written.append("gold/owner_infrastructure_exposure_snapshot.parquet")

            # Record the .duckdb artifact immediately after connect() — the file
            # is created on disk by duckdb.connect() itself, so any subsequent
            # SQL failure must still report it as written.
            con = duckdb.connect(str(gold_root / "entity_lakehouse.duckdb"))
            artifacts_written.append("gold/entity_lakehouse.duckdb")
            try:
                for name, frame in outputs.items():
                    con.execute(f"CREATE OR REPLACE TABLE dw_{name} AS SELECT * FROM frame")
                con.execute(
                    "CREATE OR REPLACE TABLE mart_owner_infrastructure_exposure_snapshot "
                    "AS SELECT * FROM owner_mart"
                )
            finally:
                con.close()
    except Exception as exc:
        # Attach whatever was written so far so callers can surface partial
        # artifact lists in failure reports even when this function raises.
        exc.__gold_artifacts__ = list(artifacts_written)  # type: ignore[attr-defined]
        raise

    outputs["owner_infrastructure_exposure_snapshot"] = owner_mart
    return outputs, artifacts_written
