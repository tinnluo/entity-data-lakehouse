"""Hybrid search over the entity master using BM25 + dense vector + RRF fusion.

Architecture
------------
BM25 leg   : bm25s (pure-Python, tunable k1/b parameters) over
             dw_entity_master_current.  Runs entirely in memory at index
             build time; no DuckDB writes are required.

Dense leg  : sentence-transformers model (all-MiniLM-L6-v2) encodes the
             query and each entity's search_text at index time.  Vectors
             are stored in a Qdrant local collection.  By default the
             collection is persisted to gold/qdrant_store/ and reused when
             the corpus/model fingerprint matches.

Fusion     : Reciprocal Rank Fusion (RRF, k=60) merges the two ranked lists
             into a single ranked result.  No raw score normalisation is
             needed because RRF operates on ranks only.

Optional L2: After fusion, results can be re-ranked with a cross-encoder
             (not wired here — left as an extension point).

Public API
----------
build_search_index(duckdb_path)   -> EntitySearchIndex
EntitySearchIndex.search(query, top_k) -> list[SearchResult]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Optional-dependency guard — bm25s, sentence-transformers, and qdrant-client
# are in the [search] extra group.  Import lazily so the rest of the pipeline
# does not break when the extra is not installed.
# ---------------------------------------------------------------------------

try:
    import bm25s  # type: ignore[import-untyped]

    _BM25S_AVAILABLE = True
except ImportError:
    _BM25S_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from qdrant_client import QdrantClient  # type: ignore[import-untyped]
    from qdrant_client.models import (  # type: ignore[import-untyped]
        Distance,
        PointStruct,
        VectorParams,
    )

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "entity_master"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension
_RRF_K = 60  # standard RRF constant from Cormack et al. 2009
_BM25_K1 = 1.5  # term-frequency saturation; 1.2–2.0 is typical
_BM25_B = 0.75  # document-length normalisation; 0.75 is the BM25 default
_FINGERPRINT_FILE = ".fingerprint"  # written inside qdrant_path after a build


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single entity returned by a hybrid search query."""

    entity_id: str
    entity_name: str
    normalized_name: str
    country_code: str
    entity_type: str
    lei: str | None
    rrf_score: float
    bm25_rank: int | None  # None if not in BM25 top-k
    vector_rank: int | None  # None if not in vector top-k


@dataclass
class EntitySearchIndex:
    """Holds the two search indexes and exposes a unified hybrid query."""

    duckdb_path: Path
    _qdrant: Any = field(repr=False, default=None)
    _embedder: Any = field(repr=False, default=None)
    _entity_rows: list[dict] = field(repr=False, default_factory=list)
    # bm25s retriever and the ordered list of corpus texts it was built on
    _bm25_retriever: Any = field(repr=False, default=None)
    _bm25_ids: list[str] = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """Return (entity_id, 1-based rank) pairs from the bm25s retriever."""
        query_tok = bm25s.tokenize([query])
        # retrieve returns (results_indices, scores) arrays of shape (n_queries, k)
        results_idx, scores = self._bm25_retriever.retrieve(
            query_tok, k=min(top_k, len(self._bm25_ids))
        )
        ranked: list[tuple[str, int]] = []
        for rank, (idx, score) in enumerate(
            zip(results_idx[0], scores[0]), start=1
        ):
            if score <= 0:
                break  # bm25s pads with zero scores when corpus < k
            ranked.append((self._bm25_ids[int(idx)], rank))
        return ranked

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """Return (entity_id, 1-based rank) pairs from Qdrant ANN search."""
        query_vec = self._embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self._qdrant.search(
            collection_name=_COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
        )
        return [(hit.payload["entity_id"], rank + 1) for rank, hit in enumerate(hits)]

    @staticmethod
    def _rrf_fuse(
        bm25_ranked: list[tuple[str, int]],
        vector_ranked: list[tuple[str, int]],
        k: int = _RRF_K,
    ) -> list[tuple[str, float, int | None, int | None]]:
        """Reciprocal Rank Fusion.

        Returns list of (entity_id, rrf_score, bm25_rank, vector_rank) sorted
        by rrf_score descending.
        """
        scores: dict[str, float] = {}
        bm25_map: dict[str, int] = {eid: r for eid, r in bm25_ranked}
        vector_map: dict[str, int] = {eid: r for eid, r in vector_ranked}

        for entity_id, rank in bm25_ranked:
            scores[entity_id] = scores.get(entity_id, 0.0) + 1.0 / (k + rank)

        for entity_id, rank in vector_ranked:
            scores[entity_id] = scores.get(entity_id, 0.0) + 1.0 / (k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (eid, score, bm25_map.get(eid), vector_map.get(eid))
            for eid, score in ranked
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Run a hybrid BM25 + dense vector query and return fused results.

        Parameters
        ----------
        query:
            Free-text search string (e.g. "solar energy Germany").
        top_k:
            Maximum number of results to return after fusion.
        """
        candidate_k = max(top_k * 4, 20)  # oversample before fusion

        bm25_hits = self._bm25_search(query, candidate_k)
        vector_hits = self._vector_search(query, candidate_k)

        fused = self._rrf_fuse(bm25_hits, vector_hits)[:top_k]

        entity_by_id = {row["entity_id"]: row for row in self._entity_rows}
        results: list[SearchResult] = []
        for entity_id, rrf_score, bm25_rank, vector_rank in fused:
            row = entity_by_id.get(entity_id)
            if row is None:
                continue
            results.append(
                SearchResult(
                    entity_id=entity_id,
                    entity_name=row["entity_name"],
                    normalized_name=row["normalized_name"],
                    country_code=row["country_code"],
                    entity_type=row["entity_type"],
                    lei=row.get("lei") or None,
                    rrf_score=round(rrf_score, 6),
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


def _build_search_text(row: dict) -> str:
    """Concatenate the fields used for both BM25 and embedding."""
    parts = [
        row.get("entity_name") or "",
        row.get("normalized_name") or "",
        row.get("country_code") or "",
        row.get("entity_type") or "",
        row.get("lei") or "",
    ]
    return " ".join(p for p in parts if p).strip()


def _compute_corpus_fingerprint(
    entity_ids: list[str],
    texts: list[str],
    model_name: str,
) -> str:
    """Return a SHA-256 hex digest of the current corpus + model name.

    Captures both entity identity *and* the full search text used for
    embedding, so any change to entity_name, normalized_name, country_code,
    entity_type, or lei will produce a different fingerprint and trigger a
    rebuild of the vector collection.
    """
    corpus_map = dict(sorted(zip(entity_ids, texts)))
    payload = json.dumps({"corpus": corpus_map, "model": model_name}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _qdrant_collection_valid(
    qdrant: "QdrantClient",
    collection_name: str,
    qdrant_path: Path,
    fingerprint: str,
) -> bool:
    """Return True iff the persisted collection exists and its fingerprint matches.

    Reads the fingerprint written by the last successful build from
    ``qdrant_path / _FINGERPRINT_FILE`` and compares it to the fingerprint
    computed from the current DuckDB rows and model name.  A mismatch means
    the corpus or model has changed and the collection must be rebuilt.
    """
    if not qdrant.collection_exists(collection_name):
        return False
    fp_path = qdrant_path / _FINGERPRINT_FILE
    if not fp_path.exists():
        return False
    return fp_path.read_text().strip() == fingerprint


def build_search_index(
    duckdb_path: Path,
    qdrant_path: Path | None = None,
) -> EntitySearchIndex:
    """Build BM25 + dense vector indexes and return an EntitySearchIndex.

    Parameters
    ----------
    duckdb_path:
        Path to entity_lakehouse.duckdb (produced by the gold pipeline stage).
    qdrant_path:
        Path for persistent Qdrant local storage.  Defaults to
        ``<duckdb_path.parent>/qdrant_store/`` so the collection survives
        restarts without re-embedding.  On each call a SHA-256 fingerprint
        of ``{entity_id: search_text}`` and the model name is compared
        against the stored fingerprint; any change to entity text fields or
        the model triggers a full rebuild, keeping BM25 and vector legs in
        sync.  Pass ``Path(":memory:")`` to force an in-memory
        (non-persistent) collection — useful in tests.
    """
    _EXTRAS_MSG = (
        "Install with: pip install 'entity-data-lakehouse[search]'"
    )
    if not _BM25S_AVAILABLE:
        raise ImportError(
            f"bm25s is required for hybrid search. {_EXTRAS_MSG}"
        )
    if not _ST_AVAILABLE:
        raise ImportError(
            f"sentence-transformers is required for hybrid search. {_EXTRAS_MSG}"
        )
    if not _QDRANT_AVAILABLE:
        raise ImportError(
            f"qdrant-client is required for hybrid search. {_EXTRAS_MSG}"
        )

    # ------------------------------------------------------------------
    # 1. Load entity master rows from DuckDB (read-only — no writes)
    # ------------------------------------------------------------------
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        entity_rows_raw = con.execute(
            "SELECT entity_id, entity_name, normalized_name, country_code, "
            "entity_type, lei FROM dw_entity_master_current"
        ).fetchall()
    finally:
        con.close()

    col_names = [
        "entity_id", "entity_name", "normalized_name",
        "country_code", "entity_type", "lei",
    ]
    entity_rows = [dict(zip(col_names, row)) for row in entity_rows_raw]
    texts = [_build_search_text(row) for row in entity_rows]
    entity_ids = [row["entity_id"] for row in entity_rows]

    # ------------------------------------------------------------------
    # 2. BM25 leg — bm25s in-memory retriever (tunable k1/b, no DB writes)
    # ------------------------------------------------------------------
    tokenized_corpus = bm25s.tokenize(texts)
    bm25_retriever = bm25s.BM25(k1=_BM25_K1, b=_BM25_B)
    bm25_retriever.index(tokenized_corpus)

    # ------------------------------------------------------------------
    # 3. Dense vector leg — sentence-transformers + Qdrant
    # ------------------------------------------------------------------
    embedder = SentenceTransformer(_EMBED_MODEL)

    # Resolve Qdrant storage path: default to <gold_dir>/qdrant_store/
    if qdrant_path is None:
        qdrant_path = duckdb_path.parent / "qdrant_store"

    # Corpus fingerprint covers entity text fields + model name.  Written to
    # disk after a successful build; checked on every subsequent call so that
    # any change to entity_name, normalized_name, country_code, entity_type,
    # or lei triggers a full rebuild rather than silently reusing stale vectors.
    fingerprint = _compute_corpus_fingerprint(entity_ids, texts, _EMBED_MODEL)

    if str(qdrant_path) == ":memory:":
        qdrant = QdrantClient(":memory:")
        _collection_ready = False
    else:
        qdrant_path.mkdir(parents=True, exist_ok=True)
        qdrant = QdrantClient(path=str(qdrant_path))
        _collection_ready = _qdrant_collection_valid(
            qdrant, _COLLECTION_NAME, qdrant_path, fingerprint
        )

    if not _collection_ready:
        # Drop stale collection if it exists (data mismatch or first-time build).
        if str(qdrant_path) != ":memory:" and qdrant.collection_exists(_COLLECTION_NAME):
            qdrant.delete_collection(_COLLECTION_NAME)
        qdrant.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
        )
        embeddings = embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={"entity_id": entity_ids[idx]},
            )
            for idx in range(len(entity_rows))
        ]
        qdrant.upsert(collection_name=_COLLECTION_NAME, points=points)
        # Write fingerprint only after a successful build.
        if str(qdrant_path) != ":memory:":
            (qdrant_path / _FINGERPRINT_FILE).write_text(fingerprint)

    return EntitySearchIndex(
        duckdb_path=duckdb_path,
        _qdrant=qdrant,
        _embedder=embedder,
        _entity_rows=entity_rows,
        _bm25_retriever=bm25_retriever,
        _bm25_ids=entity_ids,
    )
