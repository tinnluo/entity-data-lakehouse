from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .contracts import validate_dataframe
from .utils import stable_id


BRONZE_TYPED_FIELDS = {
    "source_record_id",
    "source_business_key",
    "record_type",
    "entity_name",
    "country_code",
}


def _build_bronze_records(source_name: str, snapshot_date: str, df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        typed = {key: row.get(key, "") for key in BRONZE_TYPED_FIELDS}
        raw_payload = {key: value for key, value in row.items() if key not in BRONZE_TYPED_FIELDS}
        records.append(
            {
                "load_id": stable_id("load", source_name, snapshot_date, row.get("source_record_id")),
                "snapshot_date": snapshot_date,
                "source_name": source_name,
                **typed,
                "raw_payload": json.dumps(raw_payload, ensure_ascii=True, sort_keys=True),
            }
        )
    return pd.DataFrame(records)


def ingest_sample_data(
    sample_root: Path,
    bronze_root: Path,
    contract_path: Path,
    *,
    dry_run: bool = False,
) -> dict[tuple[str, str], pd.DataFrame]:
    bronze_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for source_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        source_name = source_dir.name
        for csv_path in sorted(source_dir.glob("*.csv")):
            snapshot_date = csv_path.stem
            raw_df = pd.read_csv(csv_path, dtype=str).fillna("")
            bronze_df = _build_bronze_records(source_name, snapshot_date, raw_df)
            bronze_df["snapshot_date"] = bronze_df["snapshot_date"].astype("string")
            bronze_df["source_name"] = bronze_df["source_name"].astype("string")
            bronze_df["load_id"] = bronze_df["load_id"].astype("string")
            bronze_df["source_record_id"] = bronze_df["source_record_id"].astype("string")
            bronze_df["source_business_key"] = bronze_df["source_business_key"].astype("string")
            bronze_df["record_type"] = bronze_df["record_type"].astype("string")
            bronze_df["entity_name"] = bronze_df["entity_name"].astype("string")
            bronze_df["country_code"] = bronze_df["country_code"].astype("string")
            bronze_df["raw_payload"] = bronze_df["raw_payload"].astype("string")

            validate_dataframe(bronze_df, contract_path)

            if not dry_run:
                output_dir = bronze_root / f"source={source_name}" / f"snapshot_date={snapshot_date}"
                output_dir.mkdir(parents=True, exist_ok=True)
                bronze_df.to_parquet(output_dir / "records.parquet", index=False)
            bronze_frames[(source_name, snapshot_date)] = bronze_df
    return bronze_frames
