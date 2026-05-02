from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Container-absolute path — assumes the repo is bind-mounted at /opt/airflow/repo
# as defined in the `volumes:` section of docker-compose.yml (airflow service).
REPO_ROOT = Path("/opt/airflow/repo")


def _get_publish_mode() -> str:
    """Return the resolved publish mode for this run.

    Precedence (highest first):
    1. Airflow Variable ``PUBLISH_MODE`` (controllable from the UI).
    2. ``PUBLISH_MODE`` environment variable.
    3. Hard default: ``"commit"``.

    Allowed values: ``"commit"`` | ``"dry_run"``.

    Failure policy
    --------------
    Only ``KeyError`` (variable not set) and
    ``airflow.exceptions.AirflowNotFoundException`` (variable absent in the
    metadata DB) are treated as "variable missing" — in both cases we fall
    through to the env/default.  All other exceptions (metadata-DB errors,
    connection failures, etc.) are **re-raised** so the task fails visibly
    rather than silently degrading an intended ``dry_run`` into a mutating
    ``commit`` run.
    """
    try:
        from airflow.exceptions import AirflowNotFoundException  # noqa: PLC0415
        _not_found_exc: tuple[type[Exception], ...] = (KeyError, AirflowNotFoundException)
    except ImportError:
        _not_found_exc = (KeyError,)

    try:
        mode = Variable.get("PUBLISH_MODE", default_var=None)
    except _not_found_exc:  # type: ignore[misc]
        mode = None
    # Any other exception propagates — fail closed.
    return mode or os.environ.get("PUBLISH_MODE", "commit")


def _should_skip_dbt() -> bool:
    """Return True when dbt materialisation should be skipped.

    dbt depends on artefacts written by the pipeline (gold/entity_lakehouse.duckdb,
    gold/dw/*.parquet).  In ``dry_run`` those files are intentionally not written,
    so dbt must not run.

    Returns ``True`` when ``_get_publish_mode()`` resolves to anything other than
    ``"commit"`` (currently only ``"dry_run"`` is defined, but guard defensively).
    """
    return _get_publish_mode() != "commit"


def _run_pipeline() -> None:
    from entity_data_lakehouse.pipeline import run_pipeline
    run_pipeline(repo_root=REPO_ROOT, publish_mode=_get_publish_mode())


def _run_dbt_or_skip() -> None:
    """Run dbt in commit mode; print a clear skip message in dry_run mode."""
    import subprocess  # noqa: PLC0415
    if _should_skip_dbt():
        print("PUBLISH_MODE=dry_run: skipping dbt run (no artefacts written).")
        return
    subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd="/opt/airflow/repo/dbt",
        check=True,
    )
    subprocess.run(
        ["dbt", "test", "--profiles-dir", "."],
        cwd="/opt/airflow/repo/dbt",
        check=True,
    )


with DAG(
    dag_id="entity_lakehouse_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["entity-data-lakehouse"],
) as dag:
    run_pipeline_stages = PythonOperator(
        task_id="run_pipeline_stages",
        python_callable=_run_pipeline,
    )

    # dbt and the public-safety scan depend on artefacts produced by the pipeline
    # (gold/entity_lakehouse.duckdb, gold/dw/*.parquet).  In dry_run those files
    # are intentionally not written, so _run_dbt_or_skip() returns early (no dbt
    # execution) and run_public_safety_scan is still invoked but will find no new
    # artefacts to scan.
    run_dbt = PythonOperator(
        task_id="run_dbt",
        python_callable=_run_dbt_or_skip,
    )

    run_public_safety_scan = BashOperator(
        task_id="run_public_safety_scan",
        # Path is container-absolute; see REPO_ROOT comment above.
        bash_command="python /opt/airflow/repo/scripts/verify_public_safety.py",
    )

    run_pipeline_stages >> run_dbt >> run_public_safety_scan
