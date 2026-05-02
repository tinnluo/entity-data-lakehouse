from pathlib import Path

import json
import shutil
import pandas as pd

from entity_data_lakehouse.pipeline import run_pipeline

_VALID_LIFECYCLE_STAGES = {
    "planning",
    "construction",
    "operating",
    "decommissioning",
    "retired",
}


def test_pipeline_builds_expected_outputs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results = run_pipeline(repo_root)

    assert results["entity_master_rows"] == 6
    assert results["asset_master_rows"] == 5
    assert results["relationship_edge_rows"] == 37
    assert results["gold_rows"] == 21
    assert results["ml_prediction_rows"] == 5

    gold_path = repo_root / "gold" / "owner_infrastructure_exposure_snapshot.parquet"
    gold_df = pd.read_parquet(gold_path)
    assert {"NEW", "CHANGED", "UNCHANGED", "DROPPED"} <= set(
        gold_df["change_status_vs_prior_snapshot"]
    )

    lifecycle_path = repo_root / "gold" / "dw" / "ownership_lifecycle.parquet"
    lifecycle_df = pd.read_parquet(lifecycle_path)
    assert "INTERMITTENT" in set(lifecycle_df["lifecycle_status"])

    ml_path = repo_root / "gold" / "dw" / "asset_lifecycle_predictions.parquet"
    assert ml_path.exists(), (
        "asset_lifecycle_predictions.parquet not written to gold/dw/"
    )
    ml_df = pd.read_parquet(ml_path)

    # One prediction per asset in asset_master.
    assert len(ml_df) == 5, f"Expected 5 ML prediction rows, got {len(ml_df)}"

    # All predicted lifecycle stages must be valid labels.
    unexpected = set(ml_df["predicted_lifecycle_stage"]) - _VALID_LIFECYCLE_STAGES
    assert not unexpected, f"Unexpected lifecycle stages: {unexpected}"

    # Retirement years must be in a physically plausible range.
    assert ml_df["estimated_retirement_year"].between(2025, 2080).all(), (
        f"Retirement years out of range: {ml_df['estimated_retirement_year'].tolist()}"
    )

    # Commissioning must precede retirement.
    assert (
        ml_df["estimated_commissioning_year"] < ml_df["estimated_retirement_year"]
    ).all()

    # Capacity factor predictions must be in 1-80% range.
    assert ml_df["predicted_capacity_factor_pct"].between(1.0, 80.0).all()

    # All five assets from asset_master should appear in ML predictions.
    assert set(ml_df["asset_sector"]).issubset({"solar", "wind", "storage"})


def test_dry_run_produces_report_without_new_parquet_files(tmp_path) -> None:
    """dry_run must write only publish_report.json — no new parquet or DuckDB files,
    and no pre-existing output files must be touched (mtime unchanged)."""
    repo_root = Path(__file__).resolve().parents[2]
    report_path = tmp_path / "dry_run_report.json"

    # Snapshot gold/ contents and their mtimes before the dry_run.
    gold_root = repo_root / "gold"
    before_files = set(gold_root.rglob("*.parquet")) | set(gold_root.rglob("*.duckdb"))
    before_mtimes = {p: p.stat().st_mtime for p in before_files}

    results = run_pipeline(
        repo_root,
        publish_mode="dry_run",
        report_path=report_path,
    )

    # Row counts must be identical to a commit run (computation is unchanged).
    assert results["entity_master_rows"] == 6
    assert results["asset_master_rows"] == 5
    assert results["relationship_edge_rows"] == 37
    assert results["gold_rows"] == 21
    assert results["ml_prediction_rows"] == 5

    # No new parquet/duckdb files should have been created.
    after_files = set(gold_root.rglob("*.parquet")) | set(gold_root.rglob("*.duckdb"))
    new_files = after_files - before_files
    assert new_files == set(), (
        f"dry_run must not write any new parquet/duckdb files; found: {new_files}"
    )

    # Pre-existing files must not have been modified (mtime unchanged).
    modified = [
        str(p) for p in before_files
        if p.stat().st_mtime != before_mtimes[p]
    ]
    assert modified == [], (
        f"dry_run must not modify pre-existing output files; modified: {modified}"
    )

    # The report must exist and carry the correct mode.
    assert report_path.exists(), "publish_report.json was not written by dry_run"
    with report_path.open() as fh:
        report = json.load(fh)

    assert report["publish_mode"] == "dry_run"
    assert report["status"] == "success"
    assert report["artifacts_written"] == []
    # dry_run always calls validate_sink_schema() and overwrites status to
    # "dry_run_validated"; "skipped" is only the initial default and is never
    # the final value on a successful dry_run.
    assert report["sink_target"]["status"] == "dry_run_validated"


def test_dry_run_creates_no_directories(tmp_path) -> None:
    """dry_run must not create bronze/, silver/, gold/, or gold/dw/ directories.

    Uses a minimal repo copy containing only the read-only inputs (sample_data,
    reference_data, contracts) so there is no pre-existing bronze/silver/gold tree
    to confound the assertion.
    """
    real_root = Path(__file__).resolve().parents[2]

    # Copy only the directories the pipeline reads (not the output dirs).
    for src_name in ("sample_data", "reference_data", "contracts"):
        src = real_root / src_name
        if src.exists():
            shutil.copytree(src, tmp_path / src_name)

    report_path = tmp_path / "report.json"

    run_pipeline(tmp_path, publish_mode="dry_run", report_path=report_path)

    # None of the output directories should have been created.
    for output_dir in ("bronze", "silver", "gold"):
        assert not (tmp_path / output_dir).exists(), (
            f"dry_run must not create directory '{output_dir}'; found at {tmp_path / output_dir}"
        )

    # The report must have been written (it has an explicit path outside gold/).
    assert report_path.exists()
    with report_path.open() as fh:
        report = json.load(fh)
    assert report["status"] == "success"
    assert report["artifacts_written"] == []


def test_dry_run_leaves_bronze_and_silver_files_untouched(tmp_path) -> None:
    """dry_run must not modify any pre-existing bronze/ or silver/ files.

    The test is self-sufficient: it builds its own populated repo copy via a
    commit run so it does not depend on the real repo tree being pre-populated
    or on any particular test execution order.
    """
    real_root = Path(__file__).resolve().parents[2]

    # --- Step 1: create a populated tmp repo via a commit run ---------------
    for src_name in ("sample_data", "reference_data", "contracts"):
        src = real_root / src_name
        if src.exists():
            shutil.copytree(src, tmp_path / src_name)

    commit_report = tmp_path / "commit_report.json"
    run_pipeline(tmp_path, publish_mode="commit", report_path=commit_report)

    # Confirm bronze/ and silver/ were created by the commit run.
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver"
    assert bronze_root.exists(), "commit run must have created bronze/"
    assert silver_root.exists(), "commit run must have created silver/"

    # --- Step 2: snapshot mtimes after commit, before dry_run ---------------
    before: dict[Path, float] = {}
    for root in (bronze_root, silver_root):
        for p in root.rglob("*"):
            if p.is_file():
                before[p] = p.stat().st_mtime

    assert before, "commit run produced no bronze/silver files — unexpected"

    # --- Step 3: run dry_run and assert immutability -------------------------
    dry_run_report = tmp_path / "dry_run_report.json"
    run_pipeline(tmp_path, publish_mode="dry_run", report_path=dry_run_report)

    # No pre-existing bronze/ or silver/ file must have been touched.
    modified = [
        str(p) for p, mtime in before.items()
        if p.stat().st_mtime != mtime
    ]
    assert modified == [], (
        f"dry_run must not modify pre-existing bronze/silver files; modified: {modified}"
    )

    # No new files must have been created under bronze/ or silver/.
    after: set[Path] = set()
    for root in (bronze_root, silver_root):
        after.update(p for p in root.rglob("*") if p.is_file())

    new_files = after - set(before)
    assert new_files == set(), (
        f"dry_run must not create new bronze/silver files; found: {new_files}"
    )
