# Goals and Features — Entity Data Lakehouse

## Goal

Demonstrate a medallion-style lakehouse architecture for entity, ownership, and infrastructure asset data — showing how to model, layer, and warehouse complex relational datasets with proper SCD design and forward-looking ML analytics.

The repo shows the full structural pattern from raw source ingestion through analytics-ready gold outputs, including warehouse design choices (SCD2 vs SCD4), entity resolution, relationship modelling, and ML-based lifecycle extrapolation for climate asset datasets.

## What This Solves

Entity and ownership data arrives from multiple sources, at different grains, with conflicting schemas and irregular update cadences. This repo shows how to absorb that complexity in bronze, normalize it in silver, and produce stable, analytics-ready gold outputs — including a DuckDB analytics store — without losing source lineage or change history.

---

## Features

### Medallion Pipeline (Bronze → Silver → Gold)

Three-layer architecture with clean separation of concerns:

| Layer | Content |
|---|---|
| Bronze | Source envelopes: typed keys + `raw_payload` JSON blob preserving all unmapped attributes |
| Silver | Observation-grain tables, entity master, asset master, ownership observations, relationship edges |
| Gold | Warehouse tables (SCD4/SCD2), lifecycle metrics, analytics mart, DuckDB analytics database |

### Entity Resolution and Relationship Modelling

Silver produces canonical entity and asset dimensions with match basis and source lineage. Relationship edges model three edge types:
- `OWNS_ASSET`
- `OPERATES_ASSET`
- `PARENT_OF_ENTITY`

These edges feed directly into the ownership-attribution pipeline (see `ownership-responsibility-graph`).

### SCD4 Entity Master

Gold uses SCD4 (current + snapshot) for the entity master:
- `entity_master_current.parquet` — stable current view for consumption
- `entity_master_comprehensive_scd4.parquet` — full snapshot history
- `entity_master_event_log.parquet` — change log for audit and replay

### SCD4 + SCD2 Ownership Tables

Ownership is tracked at two levels:
- SCD4 comprehensive snapshots with source-adaptive observation grain
- `ownership_lifecycle.parquet` — separates coverage drift from genuine business change
- `ownership_current.parquet` + `ownership_history_scd2.parquet` — stable SCD2 tables for standard BI consumption

### Derived Analytics Mart

`owner_infrastructure_exposure_snapshot.parquet` — a denormalized gold mart combining entity master, ownership stakes, and asset data into a single analytics-ready snapshot for exposure analysis.

### ML-Based Asset Lifecycle Extrapolation

Implements a knowledge-encoded ML pipeline that extrapolates lifecycle and activity
attributes for infrastructure assets where ground-truth operational data is unavailable —
a common situation in energy and climate datasets sourced from public registries.

**Approach — knowledge-encoded synthetic training**

Production energy-asset lifecycle records are proprietary.  This pipeline encodes
domain expertise into a deterministic synthetic training generator (300 reference
assets, seed-fixed for reproducibility), then trains scikit-learn models on that data
and applies them to the real assets in the silver `asset_master`.  Domain knowledge
lives in two bundled reference files:

| File | Content |
|---|---|
| `reference_data/country_attributes.csv` | 29 countries × geographic and economic attributes: latitude/longitude centroid, average altitude, territorial type (coastal/inland/island/mixed), economic level, GDP tier, solar irradiance, wind speed, regulatory stability score |
| `reference_data/sector_lifecycle.csv` | Solar, wind, storage sector parameters: typical and min/max lifespan, construction and decommissioning years, base capacity factor, sensitivity coefficients for irradiance, wind, altitude, and economic level |

**Three trained models**

| Model | Algorithm | Predicts |
|---|---|---|
| `lifecycle_stage_clf` | RandomForestClassifier (200 trees) | Lifecycle stage: `planning` / `construction` / `operating` / `decommissioning` / `retired` |
| `retirement_year_reg` | GradientBoostingRegressor (200 trees) | Estimated calendar year of decommissioning |
| `capacity_factor_reg` | RandomForestRegressor (200 trees) | Annual capacity factor (%) — actual output ÷ theoretical maximum |

**Feature vector per asset**

- Asset attributes: `capacity_mw`, sector encoding
- Geographic enrichment (joined from `country_attributes.csv` by `asset_country`): latitude, longitude, average altitude, territorial type, GDP tier, solar irradiance, wind speed, regulatory stability
- Lifecycle signal from gold `ownership_lifecycle`: total appearances, presence rate, reliability score, typical sector lifespan

**Output: `gold/dw/asset_lifecycle_predictions.parquet`**

One row per asset, including all input features (for full explainability) plus:
- `predicted_lifecycle_stage` and `lifecycle_stage_confidence`
- `estimated_commissioning_year` and `estimated_retirement_year`
- `predicted_remaining_years`
- `predicted_capacity_factor_pct`
- `model_version` tag

The table is also registered in `entity_lakehouse.duckdb` as `ml_asset_lifecycle_predictions`.

**Energy industry applicability**

In production energy datasets, assets from the same country-sector combination show
strong covariance in lifespan (regulatory framework, grid standards, climate exposure)
and capacity factor (solar irradiance for PV, wind speed for turbines, altitude for
both).  The feature-to-outcome structure in this pipeline mirrors the feature engineering
used in real transition-risk and stranded-asset models.

### DuckDB Analytics Store

`entity_lakehouse.duckdb` — a local analytics database generated from gold outputs, queryable with standard SQL for ad-hoc exploration without additional infrastructure.

### Hybrid Search API and CLI

Optional BM25 + dense vector retrieval over `dw_entity_master_current` using `bm25s`, `sentence-transformers/all-MiniLM-L6-v2`, and local Qdrant. Persisted vectors live in `gold/qdrant_store/` and are reused only when the corpus/model fingerprint matches the current entity master.

### Public Safety Verification

`scripts/verify_public_safety.py` — scans for banned company references, internal paths, and credentials before any commit or publish step.

### dbt Modelling Layer

A dbt-duckdb project (`dbt/`) sits above the gold-layer DuckDB database and re-models the warehouse tables into a separate analytics schema (`main_analytics`). This leaves the upstream `main.dw_*`, `main.mart_*`, and `main.ml_*` tables produced by the Python pipeline completely untouched.

| dbt model | Source table | Target |
|---|---|---|
| `entity_master_current` | `dw_entity_master_current` | `main_analytics.entity_master_current` |
| `ownership_current` | `dw_ownership_current` | `main_analytics.ownership_current` |
| `owner_infrastructure_exposure_snapshot` | `mart_owner_infrastructure_exposure_snapshot` | `main_analytics.owner_infrastructure_exposure_snapshot` |
| `asset_lifecycle_predictions` | `ml_asset_lifecycle_predictions` | `main_analytics.asset_lifecycle_predictions` |

20 dbt tests pass (schema uniqueness/non-null + 3 singular SQL data-quality tests). See `docs/data_warehouse.md` for the full schema layout and test descriptions.

### Apache Airflow Orchestration

An Airflow 2.9 DAG (`airflow/dags/entity_lakehouse_dag.py`) wraps the full pipeline for orchestration demo purposes:

```
run_pipeline_stages  >>  run_dbt  >>  run_public_safety_scan
```

- `run_pipeline_stages` — PythonOperator: bronze → silver → gold → ML
- `run_dbt` — BashOperator: `dbt run` + `dbt test` against the gold-layer DuckDB database
- `run_public_safety_scan` — BashOperator: public-safety scan as a final gate

Uses `SequentialExecutor` + SQLite (recommended for single-machine demo). The DAG is trigger-only (`schedule=None`). See `airflow/README.md` and `docs/architecture.md` for details.

### LoRA Fine-Tuning Demo

An optional LoRA-tuned LLM path (`src/entity_data_lakehouse/ml_lora.py`) overrides the `predicted_lifecycle_stage` column when `ML_BACKEND=lora` is set. All other prediction columns (retirement year, capacity factor, etc.) continue using scikit-learn unchanged, so integration-test row counts are not affected.

All heavy dependencies (`peft`, `transformers`, `trl`, `torch`) are imported lazily inside function bodies, so the module can be imported in CI without those packages installed.

| Component | Description |
|---|---|
| `ml_lora.py` | Lazy-import module: prompt construction, JSONL generation, adapter training, inference |
| `scripts/train_lora.py` | CLI to generate synthetic JSONL and fine-tune the adapter |
| `scripts/eval_lora.py` | Accuracy / F1 / confusion matrix: LoRA vs sklearn baseline |
| `models/lifecycle_lora_adapter/` | Saved PEFT adapter weights (gitignored) |

Base model: `Qwen/Qwen2.5-0.5B-Instruct`. Hardware: MPS (Apple Silicon) or CUDA recommended.

### Docker Support

Full Docker and Docker Compose setup:

- **lakehouse service** — runs the full bronze → silver → gold → ML pipeline; generated artifacts are written back to the host via volume mounts.
- **airflow service** — runs `airflow standalone` (scheduler + webserver + triggerer) with the custom Airflow 2.9 image; UI available at `http://localhost:8080` (admin/admin).

---

## What This Repo Does Not Cover

- Document acquisition or filing retrieval (see `document-acquisition-workbench`)
- Graph-based ownership traversal and responsibility attribution (see `ownership-responsibility-graph`)
- Analytics UI over the gold outputs (see `entity-insight-studio`)
