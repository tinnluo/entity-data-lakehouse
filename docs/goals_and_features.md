# Goals and Features

This document is intentionally high level. It explains what the repo is trying to demonstrate and what features are in scope without repeating the detailed architecture, warehouse design, or runtime instructions from the other docs.

See also:

- [`../README.md`](../README.md) for the repo overview and entrypoints
- [`architecture.md`](architecture.md) for the system view and optional extension paths
- [`data_warehouse.md`](data_warehouse.md) for the SCD2/SCD4 warehouse design

## Goal

Demonstrate a public-safe, medallion-style lakehouse for entity and infrastructure ownership data that shows:

- bronze -> silver -> gold layering with explicit output contracts
- canonical entity and relationship modeling across mixed-grain public inputs
- history-aware warehouse design using both SCD4 and SCD2 patterns
- deterministic local execution with DuckDB as the default analytics store
- optional production-style extensions that do not alter the default local path

## What This Repo Solves

Entity and ownership data tends to arrive from multiple sources, at different grains, with different identifiers, irregular update cadence, and partial historical coverage. This repo shows how to absorb that mess in bronze, normalize it in silver, and publish stable, analytics-ready gold outputs without losing lineage or change history.

## Core Features

| Feature | Why it exists |
|---|---|
| Medallion pipeline | Separates ingestion, normalization, and consumption concerns so each layer has a clear contract. |
| Entity resolution + relationship modeling | Produces canonical entities, assets, and `OWNS_ASSET` / `OPERATES_ASSET` / `PARENT_OF_ENTITY` edges from heterogeneous source records. |
| Hybrid warehouse design | Uses SCD4 where full snapshot preservation matters and SCD2 where downstream consumers need stable current/history tables. |
| Derived exposure mart | Publishes a denormalized owner-to-asset exposure snapshot for analytics consumption. |
| DuckDB-first analytics store | Keeps the default path local, deterministic, and infrastructure-free. |
| ML enrichment | Adds lifecycle-stage, retirement-year, and capacity-factor predictions from a reproducible sklearn baseline. |
| Rollback-safe analytics publication | Every run emits a machine-readable `publish_report.json` with row counts, rollback status, and sink summary. `publish_mode=dry_run` validates the full pipeline and ClickHouse schemas without writing any artifacts (except the report itself). `publish_mode=commit` (default) is the full pipeline with all writes and optional ClickHouse sink. |
| Runtime- and cost-aware benchmarking | The eval harness reports quality, latency, and equivalent-cloud cost estimates for sklearn vs LoRA in a single reproducible JSON report. |
| Optional LoRA override | Allows lifecycle-stage override only, while preserving the sklearn baseline for the other prediction columns. |
| Optional extension surfaces | dbt, Airflow, ClickHouse, hybrid search, and Langfuse can be enabled without changing the default DuckDB pipeline. |
| Public-safety checks | Includes a repo scan for banned company references, secrets, and internal paths before publish. |

## Feature Boundaries

### Data modeling

The repo focuses on layered data contracts, entity matching, ownership history, and analytics-ready outputs. It is not trying to be a generic ingestion framework.

### ML

The default ML path is a deterministic sklearn baseline trained from bundled synthetic reference data. The optional LoRA path is deliberately constrained:

- it overrides only lifecycle-stage classification fields
- it is revision-pinned and base-model constrained
- it validates adapter provenance before loading
- it falls back to the sklearn outputs when validation or inference fails

### Storage

DuckDB is the source of truth. Optional ClickHouse support is a write-through sink for production-style OLAP queries, not a backend switch.

### Search

Hybrid search is an optional consumer of the current entity master. It is not part of the core bronze -> silver -> gold contract.

## Non-Goals

- document acquisition or filing retrieval workflows
- graph-style ownership traversal beyond the published warehouse outputs
- a dedicated analytics UI over the gold data
- streaming or Kafka-based ingestion
- replacing DuckDB as the authoritative local store

## Where To Read Next

- Read [`architecture.md`](architecture.md) for orchestration, ClickHouse, Airflow, ML, and observability shape.
- Read [`data_warehouse.md`](data_warehouse.md) for the detailed entity/ownership history design.
