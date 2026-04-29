# HANDOVER

## Objective

Implement the Phase 1 market-alignment upgrade for `entity-data-lakehouse`.

This is a stack-extension task, not a rewrite. The repo must keep its current strengths:

- DuckDB default
- dbt compatibility
- Airflow demo
- sklearn + LoRA comparison

The required transition is:

- keep DuckDB as the default runnable path
- add optional ClickHouse as a second analytics or serving variant
- add Langfuse instrumentation around ML and eval flows
- add a reproducible eval suite for sklearn vs LoRA

## Required outcome

When complete:

- current DuckDB pipeline still works unchanged by default
- a `USE_CLICKHOUSE=true` path exists and is documented
- ClickHouse support is optional, not mandatory for the standard demo
- ML and eval flows emit Langfuse traces when configured
- an `evals/` surface exists for repeatable sklearn vs LoRA comparison
- README and architecture docs describe the dual-path setup accurately

## Scope

In scope:

- optional ClickHouse service and plumbing
- feature flag or env-based backend selection
- ML/eval observability
- eval harness and report artifacts
- docs and tests

Out of scope:

- replacing DuckDB as the default
- introducing Kafka or full streaming in this phase
- rewriting the medallion pipeline around ClickHouse
- removing dbt or Airflow

## Implementation decisions

### 1. Keep DuckDB as the default

The standard commands must still work with no extra setup:

- `docker compose up --build`
- `python3 scripts/run_demo.py`

Default behavior remains:

- bronze -> silver -> gold pipeline
- gold outputs written as they are today
- DuckDB remains the default analytics database

### 2. Add optional ClickHouse path

Implement backend selection with:

- env var: `USE_CLICKHOUSE=true|false`
- default: `false`

ClickHouse is an optional analytics or serving variant. It does **not** replace the existing gold contract.

Required behavior:

- when `USE_CLICKHOUSE=false`, nothing regresses
- when `USE_CLICKHOUSE=true`, the repo starts a ClickHouse service and loads a clearly-defined subset of gold analytics tables or derived views into ClickHouse
- choose a small, defensible surface such as:
  - `ownership_current`
  - `owner_infrastructure_exposure_snapshot`
  - `ml_asset_lifecycle_predictions`

Do not overreach into a fake real-time architecture. No Kafka in this phase.

### 3. Add Langfuse around ML and eval runs

Instrument:

- sklearn training or evaluation
- LoRA evaluation path
- inference or artifact-generation steps where practical

Requirements:

- Langfuse is optional
- no-credential local runs still work
- local outputs and tests do not depend on remote tracing

### 4. Add evals

Create a reproducible `evals/` directory and runner covering:

- sklearn baseline quality
- LoRA quality comparison
- runtime or latency comparison
- schema or contract validation on the produced prediction artifact

The output should be machine-readable, for example:

- `evals/output/latest_report.json`

Keep metrics simple and honest. Accuracy plus runtime is enough if cost is not directly measurable.

### 5. Update docs after implementation

Update:

- `README.md`
- `docs/architecture.md`
- `airflow/README.md` only if orchestration docs need backend notes
- `.env.example`
- `docker-compose.yml`

Docs must clearly separate:

- current default DuckDB path
- optional ClickHouse path
- current batch architecture
- explicit non-goal: no streaming claim in this phase

## Likely files to modify

- `README.md`
- `docs/architecture.md`
- `docker-compose.yml`
- `.env.example`
- `pyproject.toml`
- `Makefile`
- `scripts/run_demo.py`
- `scripts/train_lora.py`
- `scripts/eval_lora.py`
- `src/entity_data_lakehouse/ml.py`
- `src/entity_data_lakehouse/ml_lora.py`
- new ClickHouse integration module under `src/entity_data_lakehouse/`
- new `evals/` directory and runner
- `tests/unit/test_ml.py`
- `tests/unit/test_ml_lora.py`
- `tests/integration/test_pipeline.py`
- new tests for ClickHouse path and eval output

## Verification

Run at minimum:

```bash
python3 -m pip install -e ".[dev]"
pytest tests/
python3 scripts/run_demo.py
python3 scripts/verify_public_safety.py
```

Verify the default container path:

```bash
docker compose up --build
docker compose run --rm lakehouse pytest tests/
```

Verify the optional ClickHouse path:

```bash
USE_CLICKHOUSE=true docker compose up --build
```

If you add a dedicated eval command or runner, execute it and confirm the report artifact is written.

Acceptance criteria:

- default DuckDB path still passes
- ClickHouse path is optional and functional
- Langfuse instrumentation does not break local runs
- eval report is reproducible
- docs match implemented reality

## Guardrails

- Do not replace DuckDB as the default
- Do not introduce Kafka or streaming claims
- Do not break dbt or Airflow paths
- Do not broaden scope into a new repo
- Keep sanitization intact

## Downstream note

This repo feeds later data-engineering and agentic CV updates. Only after ClickHouse or eval upgrades are actually shipped and verified should those skills be strengthened in the CV variants.
