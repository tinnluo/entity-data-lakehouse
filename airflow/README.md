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
| `run_pipeline_stages` | `PythonOperator` | Calls `run_pipeline(repo_root, publish_mode=PUBLISH_MODE)` to execute bronze -> silver -> gold -> ML and emit `publish_report.json` |
| `run_dbt` | `PythonOperator` | Runs `dbt run` and `dbt test` in `commit` mode; returns early with a skip message in `dry_run` (no gold artefacts to materialise) |
| `run_public_safety_scan` | `BashOperator` | Runs `scripts/verify_public_safety.py` as the final gate (both modes) |

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

### Publish Mode

The DAG reads `PUBLISH_MODE` to control safe batch publication:

| Value | Behaviour |
|---|---|
| `commit` (default) | Full pipeline with all disk writes and optional ClickHouse sink, followed by dbt run/test and public-safety scan |
| `dry_run` | Validates the full pipeline and ClickHouse schemas; writes only `publish_report.json` (and creates its parent directory if absent). The `run_dbt` task runs but returns early (no dbt execution — no gold artefacts are written, so dbt has nothing to materialise). The `run_public_safety_scan` task still runs. |

**Set via Airflow Variable (recommended — controllable from the UI):**

```bash
# In the Airflow UI: Admin → Variables → Add
# Key: PUBLISH_MODE   Value: dry_run
```

Or via CLI:

```bash
docker compose exec airflow airflow variables set PUBLISH_MODE dry_run
```

**Set via environment variable (fallback):**

```bash
PUBLISH_MODE=dry_run docker compose up airflow
```

When neither is set, the DAG defaults to `commit`.

The `publish_report.json` artifact is written to `gold/publish_report.json` after every run (both modes) and is accessible on the host via the bind-mounted repo directory.

> **Note:** In `dry_run` mode the `run_dbt` task (`PythonOperator`) returns early without invoking dbt rather than running `dbt run`. This is intentional — dbt requires the gold DuckDB file and parquet outputs that `dry_run` deliberately does not produce.

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
