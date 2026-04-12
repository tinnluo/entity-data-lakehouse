# entity-data-lakehouse

A compact medallion-style lakehouse demo that ingests public entity and infrastructure data, normalizes it into a shared entity model, and emits analytics-ready outputs.

This public demo rebuilds a production architectural pattern in a public-safe form. It preserves system design, module boundaries, and execution flow while removing proprietary business logic and internal data.

This demo shows a medallion-style entity and infrastructure pipeline, mirrors a production warehouse pattern with bronze, silver, and gold layers plus SCD2/SCD4 warehouse design, intentionally generalizes source details and removes private operating context, and exists in the portfolio to demonstrate architecture and pipeline design in a runnable public repo.

## Overview

This repository is designed as a public portfolio project. It focuses on:

- bronze, silver, and gold data layering
- entity and relationship modeling
- warehouse-oriented outputs
- deterministic local execution with bundled sample snapshots

The sample scenario combines:

- registry-style legal entities
- parent-child entity hierarchy data
- infrastructure asset ownership records

## Architecture

```text
sample_data/ CSV snapshots
        |
        v
bronze/ source envelopes with typed keys + raw_payload
        |
        v
silver/ observation tables + convenience dimensions
        |
        v
gold/ hybrid DW:
  - entity master SCD4
  - ownership SCD4 + lifecycle
  - ownership SCD2 current/history
  - derived analytics mart
        |
        v
dbt/  analytics schema (main_analytics.*) re-modelled from gold
```

See [docs/architecture.md](docs/architecture.md) and [docs/data_warehouse.md](docs/data_warehouse.md) for the detailed flow, warehouse design, and SCD2/SCD4 rationale.

## Repository Map

- `sample_data/` bundled public-safe source snapshots
- `contracts/` output contracts for bronze, silver, and gold artifacts
- `dbt/` dbt-duckdb modelling project (analytics schema on top of gold)
- `scripts/run_demo.py` single entrypoint for the local pipeline
- `scripts/verify_public_safety.py` scans for banned company references and internal paths
- `src/entity_data_lakehouse/` pipeline implementation
- `bronze/`, `silver/`, `gold/` generated run artifacts; reproducible locally and ignored by default

## Quick Start with Docker

Primary run path:

```bash
docker compose up --build
```

Equivalent legacy command:

```bash
docker-compose up --build
```

Expected result:

- the bronze -> silver -> gold pipeline runs inside the container
- generated artifacts are written to `bronze/`, `silver/`, and `gold/`
- the retained gold mart is rebuilt in `gold/owner_infrastructure_exposure_snapshot.parquet`

To run tests in the same container image:

```bash
docker compose run --rm lakehouse pytest tests/
```

## Local Run

```bash
python3 -m pip install -e '.[dev]'
python3 scripts/run_demo.py
python3 scripts/verify_public_safety.py
pytest
```

The pipeline writes generated outputs to:

- `bronze/`
- `silver/`
- `gold/`

These directories are generated artifacts for local runs. They are intentionally reproducible and ignored by default rather than treated as hand-maintained source files.

## Published Outputs

### Bronze

Each local pipeline run writes source snapshots to:

```text
bronze/source=<source>/snapshot_date=<date>/records.parquet
```

The bronze contract preserves key matching fields plus a `raw_payload` JSON blob for unmapped attributes.
In this repo, bronze outputs are generated locally from bundled sample data.

### Silver

- `entity_observations.parquet`: observation-grain entity rows for DW input
- `entity_master.parquet`: canonical entities with match basis and source lineage
- `asset_master.parquet`: infrastructure asset dimension
- `ownership_observations.parquet`: observation-grain ownership rows for DW input
- `relationship_edges.parquet`: `OWNS_ASSET`, `OPERATES_ASSET`, and `PARENT_OF_ENTITY` edges

These are generated outputs, not source-controlled assets.

### Gold

- `gold/dw/entity_master_comprehensive_scd4.parquet`
- `gold/dw/entity_master_current.parquet`
- `gold/dw/entity_master_event_log.parquet`
- `gold/dw/ownership_comprehensive_scd4.parquet`
- `gold/dw/ownership_lifecycle.parquet`
- `gold/dw/ownership_history_scd2.parquet`
- `gold/dw/ownership_current.parquet`
- `gold/owner_infrastructure_exposure_snapshot.parquet`
- `gold/entity_lakehouse.duckdb`

Gold uses a hybrid warehouse pattern inspired by the original reference repo:

- SCD4 for entity master snapshots and source-adaptive ownership observations
- lifecycle metrics to separate coverage drift from business change
- SCD2 current/history tables for stable ownership consumption
- a derived analytics mart that preserves the existing public contract

These gold artifacts are also generated locally and can be rebuilt from `sample_data/` with `python3 scripts/run_demo.py`.

## dbt Modelling Layer

A dbt-duckdb project re-models the gold-layer DuckDB tables into a separate `main_analytics` schema, leaving the upstream `main.dw_*` / `main.mart_*` / `main.ml_*` tables untouched.

```bash
pip install -e '.[dbt]'
python3 scripts/run_demo.py          # populate upstream gold tables first
make dbt-run                         # materialise main_analytics.* models
make dbt-test                        # run schema + singular data-quality tests
```

Or directly:

```bash
cd dbt && dbt run --profiles-dir . && dbt test --profiles-dir .
```

Models land in `main_analytics.*` inside `gold/entity_lakehouse.duckdb`. See [docs/data_warehouse.md](docs/data_warehouse.md) for details.

## Apache Airflow DAG

An Airflow DAG wraps the full pipeline for orchestration demo purposes, runnable via Docker:

```bash
docker compose build airflow
docker compose up airflow       # UI at http://localhost:8080 — admin/admin
make airflow-up                 # equivalent shorthand
```

The DAG `entity_lakehouse_pipeline` runs three tasks in sequence:
`run_pipeline_stages` → `run_dbt` → `run_public_safety_scan`.

Uses `SequentialExecutor` + SQLite (Airflow 2.9 recommended dev configuration).
See [airflow/README.md](airflow/README.md) and [docs/architecture.md](docs/architecture.md) for details.

## LoRA Fine-Tuning Demo

An optional LoRA-tuned LLM path overrides the `predicted_lifecycle_stage` column when
`ML_BACKEND=lora` is set.  All other predictions (retirement year, capacity factor) always
use scikit-learn, so integration-test row counts are unchanged.

**Hardware:** MPS (Apple Silicon) or CUDA recommended; CPU is slow but functional.

```bash
pip install -e '.[lora]'

# Train adapter on synthetic data (≈5 min on MPS):
python scripts/train_lora.py --samples 200 --epochs 1

# Evaluate LoRA vs sklearn baseline:
python scripts/eval_lora.py

# Run pipeline with LoRA lifecycle stage:
ML_BACKEND=lora python scripts/run_pipeline.py

# Default (ML_BACKEND unset) — identical to baseline, ml_lora never imported:
python scripts/run_pipeline.py
```

See [docs/architecture.md](docs/architecture.md) for the full design rationale.

## Public-Safety Guarantees

- This repository is a public-safe demo distilled from a production project.
- Sample data is bundled, security-safe, and intentionally small.
- No secrets, private URLs, credentials, or internal absolute paths are required to run the demo.
