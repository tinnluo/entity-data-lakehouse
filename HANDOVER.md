# HANDOVER

Repo: `entity-data-lakehouse`

## Objective

Build on the shipped lakehouse demo and make this repo the portfolio's clearest example of:

- rollback-safe analytics publication
- DuckDB-authoritative, ClickHouse-optional architecture
- ML quality / latency / cost benchmarking
- safe batch publication with explicit publish reports

This repo should remain a local-first medallion lakehouse. ClickHouse stays an optional sink, not the default backend.

## Current shipped baseline

The following are already shipped and must be preserved:

- bronze -> silver -> gold medallion pipeline
- DuckDB-authoritative gold layer
- dbt compatibility layer
- Airflow orchestration path
- optional ClickHouse write-through sink with strict schema validation
- Langfuse telemetry around ML, training, and eval paths
- sklearn baseline plus optional LoRA path
- runtime and cost-aware sklearn-vs-LoRA benchmark reporting

This handover extends the existing publish-and-benchmark story rather than redesigning it.

## Required shipped outcome

A reviewer should be able to see that this repo handles both:

- **ML workload economics** via the existing benchmark and telemetry work
- **safe analytics publication** via explicit publish-mode and rollback reporting

Minimum shipped outcome:

1. Add a publish control surface with `publish_mode=dry_run|commit`.
2. Add a machine-readable `publish_report.json`.
3. Ensure `dry_run` validates and reports without mutating ClickHouse.
4. Ensure `commit` preserves current atomic refresh and rollback behavior.
5. Surface publish behavior clearly in scripts, Airflow docs, and README.

## In scope

- publish-mode config or env surface
- machine-readable publish reporting
- dry-run validation path
- commit path preserving current behavior
- rollback visibility in artifacts
- docs and tests

## Out of scope

- making ClickHouse the default backend
- replacing DuckDB as the authoritative store
- streaming ingestion or Kafka
- redesigning the existing benchmark harness
- cloud billing integrations

## Required public framing

Use wording like:

- rollback-safe analytics publication
- DuckDB-authoritative, ClickHouse-optional
- ML quality / latency / cost benchmarking
- safe batch publication

Do not describe it as:

- a dual-primary backend architecture
- a real-time streaming analytics platform
- a ClickHouse-first repo

## Required interfaces, artifacts, and config surfaces

Add or expose:

- `publish_mode=dry_run|commit`
- artifact: `publish_report.json`

`publish_report.json` must include at least:

- run id or batch id
- publish mode
- tables attempted
- row counts
- success or failure status
- rollback status
- sink target summary

Preserve:

- DuckDB as the authoritative default path
- existing ClickHouse sink schema contracts
- existing benchmark output at `evals/output/latest_report.json`
- current local runnable path without ClickHouse

## Implementation guidance

### 1. Treat publish control as distinct from benchmarking

The repo already has ML FinOps-style benchmarking.

The new work is about publication safety and visibility:

- what would be published
- what was published
- what rolled back

### 2. Make `dry_run` real

`dry_run` must validate inputs and produce a report without mutating the sink.

It should be useful for:

- demo review
- CI validation
- preflight checks

### 3. Preserve current commit semantics

`commit` must continue to use the current strict sink contract and atomic refresh behavior.

Rollback outcomes must be visible in the publish report instead of only surfacing via logs.

### 4. Keep DuckDB authoritative in all docs

The docs must continue to make clear:

- DuckDB is the source of truth
- ClickHouse is a post-write optional serving sink

### 5. Keep ML benchmark outputs intact

Do not break the existing sklearn-vs-LoRA runtime/cost benchmark while adding publish controls.

## Likely files to modify

- `README.md`
- `docs/architecture.md`
- `airflow/README.md`
- `docker-compose.yml`
- `scripts/...`
- `src/entity_data_lakehouse/clickhouse_sink.py`
- `src/entity_data_lakehouse/pipeline.py`
- `tests/...`

## Verification commands

Run the repo's real paths after implementation. At minimum:

```bash
pytest tests/
python scripts/run_demo.py
python evals/run_evals.py
```

Add implementation-specific verification for:

- `publish_mode=dry_run` producing `publish_report.json` without sink mutation
- `publish_mode=commit` succeeding normally
- partial sink failure rolling back already-swapped tables
- rollback outcome being reflected in `publish_report.json`
- benchmark report remaining valid after publish changes

If the ClickHouse compose profile is part of the shipped path, verify that path explicitly as well.

## Guardrails

- Do not make ClickHouse mandatory for the default demo path.
- Do not weaken strict schema validation.
- Do not describe ClickHouse as a backend switch.
- Do not let `dry_run` mutate the sink.
- Do not break the existing benchmark report contract.

## Acceptance standard

Accept the implementation only if all of the following are true:

1. `publish_mode=dry_run|commit` is implemented and documented.
2. `publish_report.json` is emitted with rollback visibility.
3. `dry_run` performs validation without sink mutation.
4. `commit` preserves current atomic refresh behavior.
5. Partial sink failure still results in explicit rollback handling.
6. DuckDB remains the documented and actual authoritative path.
7. Existing ML benchmark outputs remain valid.
8. README, architecture docs, and Airflow docs describe the shipped behavior accurately.
