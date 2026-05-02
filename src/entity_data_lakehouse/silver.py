from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contracts import validate_dataframe
from .utils import normalize_name, stable_id


SOURCE_PRIORITY = {
    "registry_entities": 1,
    "entity_hierarchy": 2,
    "infrastructure_assets": 3,
}


@dataclass(frozen=True)
class Observation:
    observation_source: str
    snapshot_date: str
    source_record_id: str
    source_business_key: str
    entity_name: str
    country_code: str
    entity_type: str
    registry_entity_id: str
    lei: str
    source_entity_id: str


def _load_snapshot_frames(sample_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def read_family(family: str) -> pd.DataFrame:
        frames = []
        for csv_path in sorted((sample_root / family).glob("*.csv")):
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            df["snapshot_date"] = csv_path.stem
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    return (
        read_family("registry_entities"),
        read_family("infrastructure_assets"),
        read_family("entity_hierarchy"),
    )


def _observation_key_variants(obs: Observation) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    if obs.registry_entity_id:
        variants.append(("REGISTRY_ENTITY_ID", obs.registry_entity_id))
    if obs.lei:
        variants.append(("LEI", obs.lei))
    if obs.source_entity_id:
        variants.append(("SOURCE_ENTITY_ID", obs.source_entity_id))
    variants.append(("NAME_COUNTRY", f"{normalize_name(obs.entity_name)}|{obs.country_code}"))
    return variants


def _collect_entity_observations(
    registry_df: pd.DataFrame,
    infra_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
) -> list[Observation]:
    observations: list[Observation] = []

    for row in registry_df.to_dict(orient="records"):
        observations.append(
            Observation(
                observation_source="registry_entities",
                snapshot_date=row["snapshot_date"],
                source_record_id=row["source_record_id"],
                source_business_key=row["source_business_key"],
                entity_name=row["entity_name"],
                country_code=row["country_code"],
                entity_type=row["entity_type"],
                registry_entity_id=row["registry_entity_id"],
                lei=row["lei"],
                source_entity_id="",
            )
        )

    for row in infra_df.to_dict(orient="records"):
        observations.extend(
            [
                Observation(
                    observation_source="infrastructure_assets",
                    snapshot_date=row["snapshot_date"],
                    source_record_id=f"{row['source_record_id']}|owner",
                    source_business_key=row["source_business_key"],
                    entity_name=row["owner_name"],
                    country_code=row["owner_country_code"],
                    entity_type="company",
                    registry_entity_id="",
                    lei=row["owner_lei"],
                    source_entity_id=row["owner_source_entity_id"],
                ),
                Observation(
                    observation_source="infrastructure_assets",
                    snapshot_date=row["snapshot_date"],
                    source_record_id=f"{row['source_record_id']}|operator",
                    source_business_key=row["source_business_key"],
                    entity_name=row["operator_name"],
                    country_code=row["operator_country_code"],
                    entity_type="operator",
                    registry_entity_id="",
                    lei=row["operator_lei"],
                    source_entity_id=row["operator_source_entity_id"],
                ),
            ]
        )

    for row in hierarchy_df.to_dict(orient="records"):
        observations.extend(
            [
                Observation(
                    observation_source="entity_hierarchy",
                    snapshot_date=row["snapshot_date"],
                    source_record_id=f"{row['source_record_id']}|parent",
                    source_business_key=row["source_business_key"],
                    entity_name=row["parent_name"],
                    country_code=row["parent_country_code"],
                    entity_type="company",
                    registry_entity_id=row["parent_registry_entity_id"],
                    lei=row["parent_lei"],
                    source_entity_id=row["parent_source_entity_id"],
                ),
                Observation(
                    observation_source="entity_hierarchy",
                    snapshot_date=row["snapshot_date"],
                    source_record_id=f"{row['source_record_id']}|child",
                    source_business_key=row["source_business_key"],
                    entity_name=row["child_name"],
                    country_code=row["child_country_code"],
                    entity_type="company",
                    registry_entity_id=row["child_registry_entity_id"],
                    lei=row["child_lei"],
                    source_entity_id=row["child_source_entity_id"],
                ),
            ]
        )

    return sorted(
        observations,
        key=lambda obs: (
            obs.snapshot_date,
            SOURCE_PRIORITY[obs.observation_source],
            obs.source_record_id,
        ),
    )


def _resolve_entities(
    observations: list[Observation],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], str]]:
    identifier_index: dict[tuple[str, str], str] = {}
    entity_rows: list[dict] = []
    entity_rollup: dict[str, dict] = {}

    for obs in observations:
        variants = _observation_key_variants(obs)
        entity_id = ""
        matched_via = ""
        for key_name, key_value in variants:
            entity_id = identifier_index.get((key_name, key_value), "")
            if entity_id:
                matched_via = key_name
                break
        if not entity_id:
            primary_name, primary_value = variants[0]
            entity_id = stable_id("ent", primary_name, primary_value)
            matched_via = "INITIAL_ASSIGNMENT"

        for key_name, key_value in variants:
            identifier_index[(key_name, key_value)] = entity_id

        strongest_name, strongest_value = variants[0]
        observation_key = f"{strongest_name}:{strongest_value}"
        observation_hash = stable_id("obsh", observation_key, obs.observation_source)

        entity_rows.append(
            {
                "entity_observation_id": stable_id(
                    "eobs",
                    obs.snapshot_date,
                    obs.observation_source,
                    obs.source_record_id,
                ),
                "entity_id": entity_id,
                "observation_key": observation_key,
                "observation_key_hash": observation_hash,
                "entity_name": obs.entity_name,
                "normalized_name": normalize_name(obs.entity_name),
                "country_code": obs.country_code,
                "entity_type": obs.entity_type,
                "registry_entity_id": obs.registry_entity_id,
                "lei": obs.lei,
                "source_entity_id": obs.source_entity_id,
                "observation_source": obs.observation_source,
                "source_record_id": obs.source_record_id,
                "source_business_key": obs.source_business_key,
                "snapshot_date": obs.snapshot_date,
                "source_version": obs.snapshot_date,
                "entity_resolution_method": matched_via,
                "source_priority": SOURCE_PRIORITY[obs.observation_source],
            }
        )

        rollup = entity_rollup.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_name": obs.entity_name,
                "normalized_name": normalize_name(obs.entity_name),
                "country_code": obs.country_code,
                "entity_type": obs.entity_type,
                "registry_entity_id": obs.registry_entity_id,
                "lei": obs.lei,
                "source_systems": set(),
                "match_basis": strongest_name.casefold(),
                "first_seen_snapshot": obs.snapshot_date,
                "last_seen_snapshot": obs.snapshot_date,
            },
        )
        rollup["source_systems"].add(obs.observation_source)
        rollup["first_seen_snapshot"] = min(rollup["first_seen_snapshot"], obs.snapshot_date)
        rollup["last_seen_snapshot"] = max(rollup["last_seen_snapshot"], obs.snapshot_date)
        if obs.registry_entity_id and not rollup["registry_entity_id"]:
            rollup["registry_entity_id"] = obs.registry_entity_id
        if obs.lei and not rollup["lei"]:
            rollup["lei"] = obs.lei
        if len(obs.entity_name) > len(rollup["entity_name"]):
            rollup["entity_name"] = obs.entity_name
            rollup["normalized_name"] = normalize_name(obs.entity_name)

    latest_snapshot = max(obs.snapshot_date for obs in observations)
    entity_master = pd.DataFrame(
        [
            {
                **row,
                "source_systems": "|".join(sorted(row["source_systems"])),
                "is_current": row["last_seen_snapshot"] == latest_snapshot,
            }
            for row in entity_rollup.values()
        ]
    ).sort_values(["entity_name", "entity_id"]).reset_index(drop=True)
    entity_master["is_current"] = entity_master["is_current"].astype("bool")

    entity_observations = pd.DataFrame(entity_rows).sort_values(
        ["snapshot_date", "observation_source", "entity_id", "source_record_id"]
    ).reset_index(drop=True)
    entity_observations["source_priority"] = entity_observations["source_priority"].astype("int64")
    return entity_observations, entity_master, identifier_index


def _build_asset_master(infra_df: pd.DataFrame, entity_lookup: dict[tuple[str, str], str]) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    latest_snapshot = infra_df["snapshot_date"].max()
    for row in infra_df.to_dict(orient="records"):
        asset_id = stable_id("ast", row["asset_source_id"])
        rows.setdefault(
            asset_id,
            {
                "asset_id": asset_id,
                "asset_name": row["asset_name"],
                "asset_country": row["asset_country"],
                "asset_sector": row["asset_sector"],
                "capacity_mw": float(row["capacity_mw"]),
                "operator_entity_id": entity_lookup.get(("SOURCE_ENTITY_ID", row["operator_source_entity_id"]), ""),
                "source_systems": set(),
                "first_seen_snapshot": row["snapshot_date"],
                "last_seen_snapshot": row["snapshot_date"],
            },
        )
        rows[asset_id]["source_systems"].add("infrastructure_assets")
        rows[asset_id]["first_seen_snapshot"] = min(rows[asset_id]["first_seen_snapshot"], row["snapshot_date"])
        rows[asset_id]["last_seen_snapshot"] = max(rows[asset_id]["last_seen_snapshot"], row["snapshot_date"])

    asset_master = pd.DataFrame(
        [
            {
                **row,
                "source_systems": "|".join(sorted(row["source_systems"])),
                "is_current": row["last_seen_snapshot"] == latest_snapshot,
            }
            for row in rows.values()
        ]
    ).sort_values(["asset_name", "asset_id"]).reset_index(drop=True)
    asset_master["is_current"] = asset_master["is_current"].astype("bool")
    return asset_master


def _build_ownership_observations(
    infra_df: pd.DataFrame,
    entity_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in infra_df.to_dict(orient="records"):
        owner_entity_id = entity_lookup[("SOURCE_ENTITY_ID", row["owner_source_entity_id"])]
        asset_id = stable_id("ast", row["asset_source_id"])
        lifecycle_key = f"{owner_entity_id}|{asset_id}"
        observation_key = f"{lifecycle_key}|infrastructure_assets"
        rounded_pct = round(float(row["ownership_pct"]), 2)
        rows.append(
            {
                "ownership_observation_id": stable_id("oobs", row["snapshot_date"], row["source_record_id"]),
                "owner_entity_id": owner_entity_id,
                "owner_entity_name": row["owner_name"],
                "asset_id": asset_id,
                "asset_name": row["asset_name"],
                "asset_country": row["asset_country"],
                "asset_sector": row["asset_sector"],
                "capacity_mw": float(row["capacity_mw"]),
                "ownership_pct": rounded_pct,
                "lifecycle_key": lifecycle_key,
                "observation_key": observation_key,
                "observation_key_hash": stable_id("ohsh", observation_key),
                "observation_source": "infrastructure_assets",
                "source_record_id": row["source_record_id"],
                "source_business_key": row["source_business_key"],
                "snapshot_date": row["snapshot_date"],
                "observation_date": row["snapshot_date"],
                "source_version": row["snapshot_date"],
                "row_hash": stable_id(
                    "orow",
                    owner_entity_id,
                    asset_id,
                    row["asset_name"],
                    row["asset_country"],
                    row["asset_sector"],
                    rounded_pct,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["snapshot_date", "owner_entity_id", "asset_id", "source_record_id"]
    ).reset_index(drop=True)


def _build_relationship_edges(
    infra_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    entity_lookup: dict[tuple[str, str], str],
) -> pd.DataFrame:
    edge_rows: list[dict] = []
    seen_operator_edges: set[tuple[str, str, str]] = set()

    for row in infra_df.to_dict(orient="records"):
        asset_id = stable_id("ast", row["asset_source_id"])
        owner_entity_id = entity_lookup[("SOURCE_ENTITY_ID", row["owner_source_entity_id"])]
        operator_entity_id = entity_lookup[("SOURCE_ENTITY_ID", row["operator_source_entity_id"])]

        edge_rows.append(
            {
                "relationship_id": stable_id("rel", row["snapshot_date"], owner_entity_id, asset_id, "OWNS_ASSET"),
                "from_node_type": "entity",
                "from_node_id": owner_entity_id,
                "to_node_type": "asset",
                "to_node_id": asset_id,
                "relationship_type": "OWNS_ASSET",
                "ownership_pct": round(float(row["ownership_pct"]), 2),
                "asset_role": "owner",
                "source_name": "infrastructure_assets",
                "snapshot_date": row["snapshot_date"],
                "evidence_key": row["source_business_key"],
            }
        )

        operator_edge_key = (row["snapshot_date"], operator_entity_id, asset_id)
        if operator_edge_key not in seen_operator_edges:
            edge_rows.append(
                {
                    "relationship_id": stable_id("rel", row["snapshot_date"], operator_entity_id, asset_id, "OPERATES_ASSET"),
                    "from_node_type": "entity",
                    "from_node_id": operator_entity_id,
                    "to_node_type": "asset",
                    "to_node_id": asset_id,
                    "relationship_type": "OPERATES_ASSET",
                    "ownership_pct": 100.0,
                    "asset_role": "operator",
                    "source_name": "infrastructure_assets",
                    "snapshot_date": row["snapshot_date"],
                    "evidence_key": stable_id("evidence", row["snapshot_date"], operator_entity_id, asset_id),
                }
            )
            seen_operator_edges.add(operator_edge_key)

    for row in hierarchy_df.to_dict(orient="records"):
        parent_id = entity_lookup[("REGISTRY_ENTITY_ID", row["parent_registry_entity_id"])]
        child_id = entity_lookup[("REGISTRY_ENTITY_ID", row["child_registry_entity_id"])]
        edge_rows.append(
            {
                "relationship_id": stable_id("rel", row["snapshot_date"], parent_id, child_id, "PARENT_OF_ENTITY"),
                "from_node_type": "entity",
                "from_node_id": parent_id,
                "to_node_type": "entity",
                "to_node_id": child_id,
                "relationship_type": "PARENT_OF_ENTITY",
                "ownership_pct": 100.0,
                "asset_role": "parent",
                "source_name": "entity_hierarchy",
                "snapshot_date": row["snapshot_date"],
                "evidence_key": row["source_business_key"],
            }
        )

    edges = pd.DataFrame(edge_rows).sort_values(
        ["snapshot_date", "relationship_type", "relationship_id"]
    ).reset_index(drop=True)

    prior_signatures: set[tuple[str, str, str, str]] = set()
    statuses: list[str] = []
    for row in edges.to_dict(orient="records"):
        signature = (
            row["from_node_id"],
            row["to_node_id"],
            row["relationship_type"],
            str(row["ownership_pct"]),
        )
        statuses.append("UNCHANGED" if signature in prior_signatures else "NEW")
        prior_signatures.add(signature)
    edges["change_status"] = statuses
    return edges


def build_silver_outputs(
    sample_root: Path,
    silver_root: Path,
    contract_paths: dict[str, Path],
    *,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    registry_df, infra_df, hierarchy_df = _load_snapshot_frames(sample_root)
    observations = _collect_entity_observations(registry_df, infra_df, hierarchy_df)
    entity_observations, entity_master, entity_lookup = _resolve_entities(observations)
    asset_master = _build_asset_master(infra_df, entity_lookup)
    ownership_observations = _build_ownership_observations(infra_df, entity_lookup)
    relationship_edges = _build_relationship_edges(infra_df, hierarchy_df, entity_lookup)

    outputs = {
        "entity_observations": entity_observations,
        "entity_master": entity_master,
        "asset_master": asset_master,
        "ownership_observations": ownership_observations,
        "relationship_edges": relationship_edges,
    }

    if not dry_run:
        silver_root.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        validate_dataframe(frame, contract_paths[name])
        if not dry_run:
            frame.to_parquet(silver_root / f"{name}.parquet", index=False)
    return outputs
