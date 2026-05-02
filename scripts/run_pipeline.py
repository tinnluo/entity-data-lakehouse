from __future__ import annotations

import argparse
from pathlib import Path

from entity_data_lakehouse import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the entity-data-lakehouse pipeline.",
    )
    parser.add_argument(
        "--publish-mode",
        choices=["commit", "dry_run"],
        default="commit",
        help=(
            "commit (default): full pipeline with all disk writes and optional "
            "ClickHouse sink. "
            "dry_run: validate and report; skips all pipeline disk writes and sink "
            "mutations (publish_report.json is the only file written)."
        ),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help=(
            "Where to write publish_report.json. "
            "Defaults to {repo_root}/gold/publish_report.json."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results = run_pipeline(
        repo_root,
        publish_mode=args.publish_mode,
        report_path=args.report_path,
    )
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
