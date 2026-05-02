"""Unconditional syntax check for the Airflow DAG file.

Runs in *all* CI environments (no apache-airflow required).

Scope
-----
``py_compile.compile`` catches **SyntaxError** only — it parses the source
without executing it, so import-time ``NameError`` (e.g. missing imports) are
**not** detected here.  Those regressions are caught by the Airflow-enabled
behavioral tests in ``test_airflow_dag_import.py`` (skipped when
apache-airflow is not installed).

This test exists as a fast, dependency-free first line of defence against
obvious source breakage.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

DAG_FILE = (
    Path(__file__).resolve().parents[2]
    / "airflow"
    / "dags"
    / "entity_lakehouse_dag.py"
)


def test_dag_file_has_no_syntax_errors() -> None:
    """The DAG file must parse without SyntaxError in every CI environment.

    Note: ``py_compile`` does **not** execute the module, so it will not catch
    import-time ``NameError`` or missing-import regressions.  Those are covered
    by ``test_airflow_dag_import.py`` when apache-airflow is installed.
    """
    assert DAG_FILE.exists(), f"DAG file not found: {DAG_FILE}"
    py_compile.compile(str(DAG_FILE), doraise=True)
