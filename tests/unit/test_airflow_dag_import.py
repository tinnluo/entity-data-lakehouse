"""Smoke-test that the Airflow DAG file can be imported and has the correct dag_id.

This test is skipped when `apache-airflow` is not installed, so base CI (which does not
install the airflow extra) stays green.

An unconditional syntax check lives in ``test_airflow_dag_syntax.py`` and runs in all
CI environments regardless of whether apache-airflow is installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DAG_FILE = (
    Path(__file__).resolve().parents[2]
    / "airflow"
    / "dags"
    / "entity_lakehouse_dag.py"
)

# Guard: skip unless the real apache-airflow package (not the local airflow/ namespace)
# is present. We check for airflow.DAG which only exists in the real package.
airflow_dag = pytest.importorskip(
    "airflow.models.dag", reason="apache-airflow not installed"
)


def _load_dag_module():
    assert DAG_FILE.exists(), f"DAG file not found: {DAG_FILE}"
    spec = importlib.util.spec_from_file_location("entity_lakehouse_dag", DAG_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_dag_imports() -> None:
    module = _load_dag_module()
    assert module.dag.dag_id == "entity_lakehouse_pipeline"


def test_get_publish_mode_defaults_to_commit(monkeypatch) -> None:
    """_get_publish_mode() must return 'commit' when no variable or env var is set."""
    monkeypatch.delenv("PUBLISH_MODE", raising=False)
    module = _load_dag_module()
    # Patch Variable.get to simulate no Airflow Variable set.
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None):
        assert module._get_publish_mode() == "commit"


def test_get_publish_mode_reads_env_var(monkeypatch) -> None:
    """_get_publish_mode() must return the PUBLISH_MODE env var when no Airflow Variable is set."""
    monkeypatch.setenv("PUBLISH_MODE", "dry_run")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None):
        assert module._get_publish_mode() == "dry_run"


def test_get_publish_mode_airflow_variable_takes_precedence(monkeypatch) -> None:
    """Airflow Variable must take precedence over the PUBLISH_MODE env var."""
    monkeypatch.setenv("PUBLISH_MODE", "commit")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value="dry_run"):
        assert module._get_publish_mode() == "dry_run"


def test_should_skip_dbt_returns_true_in_dry_run(monkeypatch) -> None:
    """_should_skip_dbt() must return True when publish mode is dry_run."""
    monkeypatch.setenv("PUBLISH_MODE", "dry_run")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None):
        assert module._should_skip_dbt() is True


def test_should_skip_dbt_returns_false_in_commit(monkeypatch) -> None:
    """_should_skip_dbt() must return False when publish mode is commit."""
    monkeypatch.setenv("PUBLISH_MODE", "commit")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None):
        assert module._should_skip_dbt() is False


def test_should_skip_dbt_airflow_variable_overrides_env(monkeypatch) -> None:
    """Airflow Variable must override the env var for the skip decision."""
    monkeypatch.setenv("PUBLISH_MODE", "commit")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value="dry_run"):
        assert module._should_skip_dbt() is True


def test_run_dbt_task_is_python_operator() -> None:
    """run_dbt must be a PythonOperator (not BashOperator) so the skip logic is testable."""
    from airflow.operators.python import PythonOperator
    module = _load_dag_module()
    task = module.dag.get_task("run_dbt")
    assert isinstance(task, PythonOperator), (
        f"run_dbt should be a PythonOperator, got {type(task).__name__}"
    )


def test_run_dbt_or_skip_skips_when_dry_run(monkeypatch) -> None:
    """_run_dbt_or_skip() must return early without calling subprocess in dry_run."""
    monkeypatch.setenv("PUBLISH_MODE", "dry_run")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None), \
         patch("subprocess.run") as mock_run:
        module._run_dbt_or_skip()
        mock_run.assert_not_called()


def test_run_dbt_or_skip_calls_dbt_in_commit(monkeypatch) -> None:
    """_run_dbt_or_skip() must call dbt run and dbt test in commit mode."""
    monkeypatch.setenv("PUBLISH_MODE", "commit")
    module = _load_dag_module()
    from unittest.mock import patch
    with patch.object(module.Variable, "get", return_value=None), \
         patch("subprocess.run") as mock_run:
        module._run_dbt_or_skip()
        assert mock_run.call_count == 2
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert any("dbt" in str(c) and "run" in str(c) for c in calls)
        assert any("dbt" in str(c) and "test" in str(c) for c in calls)


def test_get_publish_mode_reraises_on_backend_error(monkeypatch) -> None:
    """_get_publish_mode() must NOT swallow unexpected Variable backend errors.

    Only KeyError / AirflowNotFoundException (variable absent) should be
    treated as 'not set'.  Any other exception must propagate so the task
    fails visibly rather than silently degrading dry_run → commit.
    """
    monkeypatch.delenv("PUBLISH_MODE", raising=False)
    module = _load_dag_module()
    from unittest.mock import patch
    import pytest as _pytest

    class _BackendError(RuntimeError):
        """Simulates a transient Airflow metadata-DB failure."""

    with patch.object(module.Variable, "get", side_effect=_BackendError("db down")):
        with _pytest.raises(_BackendError):
            module._get_publish_mode()
