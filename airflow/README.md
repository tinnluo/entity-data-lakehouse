# Airflow Local Dev Guide

This directory contains the Airflow DAG that wraps the lakehouse pipeline for orchestration demo purposes.

This document is intentionally narrow in scope: it explains how the Airflow wrapper behaves and how to run it locally. It does not repeat the full repo architecture or warehouse design.

See also:

- [`../README.md`](../README.md) for repo-level setup and optional features
- [`../docs/architecture.md`](../docs/architecture.md) for the system view and DAG placement

## What The DAG Does

The DAG `entity_lakehouse_pipeline` is a thin orchestration wrapper around the existing local pipeline.

```text
run_pipeline_stages  >>  run_dbt  >>  run_public_safety_scan
```

| Task | Operator | Purpose |
|---|---|---|
| `run_pipeline_stages` | `PythonOperator` | Calls `run_pipeline(repo_root)` to execute bronze -> silver -> gold -> ML |
| `run_dbt` | `BashOperator` | Runs `dbt run` and `dbt test` against the DuckDB-backed gold outputs |
| `run_public_safety_scan` | `BashOperator` | Runs `scripts/verify_public_safety.py` as the final gate |

Important design point: the DAG does not implement a second copy of pipeline logic. It orchestrates the same Python pipeline and dbt project used elsewhere in the repo.

## Runtime Model

- trigger-only DAG: `schedule=None`
- executor: `SequentialExecutor`
- metadata DB: SQLite
- Airflow mode: `airflow standalone`

This is a deliberate single-machine demo configuration, not a production Airflow deployment pattern.

## Before You Start

Set credentials in the repo-root `.env` file:

```bash
AIRFLOW_ADMIN_USER=admin
# AIRFLOW_ADMIN_PASSWORD=...   # required
```

The compose file refuses to start the Airflow service if `AIRFLOW_ADMIN_PASSWORD` is unset or empty.

## Run Locally

```bash
docker compose build airflow
docker compose up airflow

# or via Makefile
make airflow-up
```

UI: `http://localhost:8080`

Login with the values from `.env`:

- `AIRFLOW_ADMIN_USER`
- `AIRFLOW_ADMIN_PASSWORD`

Trigger the DAG from the UI, or from the CLI:

```bash
docker compose exec airflow airflow dags trigger entity_lakehouse_pipeline
```

## Stop

```bash
docker compose down

# or via Makefile
make airflow-down
```

## Repo Integration Notes

- the whole repo is bind-mounted at `/opt/airflow/repo`
- `PYTHONPATH=/opt/airflow/repo/src` makes `entity_data_lakehouse` importable inside the container
- generated artifacts written by the DAG appear on the host in `bronze/`, `silver/`, and `gold/`
- dbt runs against the same `gold/entity_lakehouse.duckdb` file produced by the pipeline task

## Optional Features During Airflow Runs

### ClickHouse

Set `USE_CLICKHOUSE=true` in `.env` to enable the optional ClickHouse sink during Airflow-triggered runs.

If you do that, the ClickHouse service must also be running:

```bash
make clickhouse-up
```

The compose file health-gates Airflow on ClickHouse readiness when the `clickhouse` profile is active.

### Langfuse

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to forward ML, training, and eval traces to Langfuse.

When those variables are absent, tracing degrades to the repo's no-op client and the tasks keep running normally.

## Non-Goals

- production-grade Airflow deployment guidance
- Celery/Kubernetes executor setup
- remote metadata database configuration
- duplicating the main pipeline, dbt, or observability docs here
