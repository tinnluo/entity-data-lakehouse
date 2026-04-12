from __future__ import annotations

import re
from pathlib import Path


_FA = "F" + "A"
_FORWARD = "Forward"
_ANALYTICS = "Analytics"
_PIPELINE = "data" + "_" + "pipeline"

BANNED_TOKENS = [
    f"{_FORWARD} {_ANALYTICS}",
    (_FORWARD + _ANALYTICS).casefold(),
    f"{_FA}_{_PIPELINE}",
    "f" + "a" + "_",
]


def scan_public_safety(repo_root: Path) -> list[str]:
    findings: list[str] = []
    generic_internal_path_patterns = [
        re.compile("/" + "Users" + r"/[^/\n]+/" + "Documents" + r"/[^/\n]+/"),
        re.compile("/" + "home" + r"/[^/\n]+/[^/\n]+/"),
    ]
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".parquet", ".duckdb", ".pyc"}:
            continue
        if any(
            part
            in {
                ".ruff_cache",
                ".pytest_cache",
                "__pycache__",
                ".venv",
                "node_modules",
                ".opencode",
                "dbt_packages",
            }
            for part in path.parts
        ):
            continue
        # Skip dbt-generated artefact directories (contain machine-written absolute paths).
        if any(
            len(path.parts) > i + 1
            and path.parts[i] == "dbt"
            and path.parts[i + 1] in {"target", "logs"}
            for i in range(len(path.parts) - 1)
        ):
            continue
        if any(part.endswith(".egg-info") for part in path.parts):
            continue
        if path.name == "public_safety.py":
            continue
        text = path.read_text(errors="ignore")
        for token in BANNED_TOKENS:
            if token in text:
                findings.append(
                    f"{path.relative_to(repo_root)} contains banned token '{token}'"
                )
        if any(pattern.search(text) for pattern in generic_internal_path_patterns):
            findings.append(
                f"{path.relative_to(repo_root)} contains an internal absolute path pattern"
            )
    return findings
