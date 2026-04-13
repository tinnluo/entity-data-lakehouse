"""FastAPI service exposing the hybrid entity search index as a REST endpoint.

Start the server
----------------
    uvicorn entity_data_lakehouse.api:app --reload --port 8000

    # or via the helper script:
    python scripts/search_demo.py --serve

Endpoints
---------
    GET  /health             liveness check
    GET  /search?q=...&top_k=5   hybrid search

Example
-------
    curl "http://localhost:8000/search?q=solar+Germany&top_k=3"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    import uvicorn  # noqa: F401 – imported to surface missing-dep early
except ImportError as exc:
    raise ImportError(
        "FastAPI / uvicorn are required to run the API server. "
        "Install with: pip install 'entity-data-lakehouse[search]'"
    ) from exc

from .search import EntitySearchIndex, SearchResult, build_search_index

# ---------------------------------------------------------------------------
# Application-level state: index is built once at startup.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Entity Hybrid Search API",
    description=(
        "bm25s + dense vector (sentence-transformers + Qdrant) "
        "hybrid search over the entity master, fused with Reciprocal Rank Fusion."
    ),
    version="0.1.0",
)

_index: EntitySearchIndex | None = None


def _get_duckdb_path() -> Path:
    env_path = os.environ.get("ENTITY_DUCKDB_PATH")
    if env_path:
        return Path(env_path)
    # Default: repo root is two levels above this file (src/entity_data_lakehouse/)
    return Path(__file__).resolve().parents[2] / "gold" / "entity_lakehouse.duckdb"


@app.on_event("startup")
def _startup() -> None:
    global _index
    duckdb_path = _get_duckdb_path()
    if not duckdb_path.exists():
        # Non-fatal at startup — /search will return 503.
        return
    _index = build_search_index(duckdb_path)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "index": "ready" if _index is not None else "not_loaded"}


@app.get("/search")
def search(
    q: Annotated[str, Query(min_length=1, description="Search query string")],
    top_k: Annotated[int, Query(ge=1, le=50, description="Number of results")] = 5,
) -> JSONResponse:
    """Hybrid BM25 + dense vector search over the entity master."""
    if _index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Search index not loaded. "
                "Ensure the pipeline has been run and ENTITY_DUCKDB_PATH is set "
                "or gold/entity_lakehouse.duckdb exists."
            ),
        )
    results: list[SearchResult] = _index.search(q, top_k=top_k)
    return JSONResponse(
        content={
            "query": q,
            "top_k": top_k,
            "count": len(results),
            "results": [
                {
                    "rank": rank,
                    "entity_id": r.entity_id,
                    "entity_name": r.entity_name,
                    "normalized_name": r.normalized_name,
                    "country_code": r.country_code,
                    "entity_type": r.entity_type,
                    "lei": r.lei,
                    "rrf_score": r.rrf_score,
                    "bm25_rank": r.bm25_rank,
                    "vector_rank": r.vector_rank,
                }
                for rank, r in enumerate(results, start=1)
            ],
        }
    )
