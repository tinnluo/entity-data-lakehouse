"""Unit tests for the hybrid search module.

Test strategy
-------------
- RRF fusion logic: pure Python, no external dependencies — always runs.
- _build_search_text helper: pure Python — always runs.
- Index build + search: requires [search] extras (bm25s, sentence-transformers,
  qdrant-client).  Tests are skipped automatically when the extras are absent
  via pytest.mark.skipif.  Uses a real (tiny) gold-layer DuckDB file produced by
  the integration pipeline fixture, so the test verifies end-to-end behaviour.

Fixture pattern
---------------
Follows the existing scope="module" pattern from test_ml.py: the
gold-layer DuckDB file is built once per module run via the integration
pipeline and reused across all search tests.

Qdrant is forced to in-memory mode (qdrant_path=Path(":memory:")) so the
fixture is fully self-contained and leaves no on-disk state between runs.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Availability flags (evaluated once at import time, no importorskip so the
# pure-Python tests below always run).
# ---------------------------------------------------------------------------

_BM25S_AVAILABLE = importlib.util.find_spec("bm25s") is not None
_ST_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_QDRANT_AVAILABLE = importlib.util.find_spec("qdrant_client") is not None
_SEARCH_EXTRAS = _BM25S_AVAILABLE and _ST_AVAILABLE and _QDRANT_AVAILABLE

_requires_search = pytest.mark.skipif(
    not _SEARCH_EXTRAS,
    reason=(
        "bm25s, sentence-transformers, and qdrant-client not installed; "
        "skipping search integration tests."
    ),
)

# ---------------------------------------------------------------------------
# Pure-Python tests — no extra dependencies required.
# ---------------------------------------------------------------------------


def test_rrf_fuse_basic() -> None:
    """RRF merges two lists and penalises lower ranks."""
    from entity_data_lakehouse.search import EntitySearchIndex

    bm25 = [("A", 1), ("B", 2), ("C", 3)]
    vec = [("B", 1), ("C", 2), ("A", 3)]
    fused = EntitySearchIndex._rrf_fuse(bm25, vec)

    entity_ids = [f[0] for f in fused]
    scores = [f[1] for f in fused]

    # All three entities must appear
    assert set(entity_ids) == {"A", "B", "C"}
    # Scores must be descending
    assert scores == sorted(scores, reverse=True)
    # B ranks 2nd in BM25 and 1st in vector → highest combined score
    assert entity_ids[0] == "B"


def test_rrf_fuse_exclusive_hits() -> None:
    """Entities appearing in only one list still get a score."""
    from entity_data_lakehouse.search import EntitySearchIndex

    bm25 = [("X", 1)]
    vec = [("Y", 1)]
    fused = EntitySearchIndex._rrf_fuse(bm25, vec)

    entity_ids = [f[0] for f in fused]
    assert "X" in entity_ids
    assert "Y" in entity_ids
    # Both rank 1 in their own list → equal RRF score
    scores = {f[0]: f[1] for f in fused}
    assert abs(scores["X"] - scores["Y"]) < 1e-9


def test_rrf_fuse_k_constant() -> None:
    """Higher k flattens score differences between ranks."""
    from entity_data_lakehouse.search import EntitySearchIndex

    bm25 = [("A", 1), ("B", 10)]
    vec: list[tuple[str, int]] = []

    fused_low_k = EntitySearchIndex._rrf_fuse(bm25, vec, k=1)
    fused_high_k = EntitySearchIndex._rrf_fuse(bm25, vec, k=1000)

    score_low = {f[0]: f[1] for f in fused_low_k}
    score_high = {f[0]: f[1] for f in fused_high_k}

    diff_low = score_low["A"] - score_low["B"]
    diff_high = score_high["A"] - score_high["B"]
    assert diff_low > diff_high  # low k amplifies rank differences


def test_rrf_fuse_bm25_vec_ranks_stored() -> None:
    """Fused results carry original rank provenance."""
    from entity_data_lakehouse.search import EntitySearchIndex

    bm25 = [("A", 1), ("B", 2)]
    vec = [("A", 2), ("B", 1)]
    fused = EntitySearchIndex._rrf_fuse(bm25, vec)

    by_id = {f[0]: f for f in fused}
    assert by_id["A"][2] == 1   # bm25_rank for A
    assert by_id["A"][3] == 2   # vector_rank for A
    assert by_id["B"][2] == 2   # bm25_rank for B
    assert by_id["B"][3] == 1   # vector_rank for B


def test_build_search_text_combines_fields() -> None:
    from entity_data_lakehouse.search import _build_search_text

    row = {
        "entity_name": "Acme Solar",
        "normalized_name": "acmesolar",
        "country_code": "DE",
        "entity_type": "OPERATOR",
        "lei": "XYZABC123",
    }
    text = _build_search_text(row)
    assert "Acme Solar" in text
    assert "DE" in text
    assert "OPERATOR" in text
    assert "XYZABC123" in text


def test_build_search_text_handles_none_fields() -> None:
    from entity_data_lakehouse.search import _build_search_text

    row = {
        "entity_name": "Acme",
        "normalized_name": None,
        "country_code": "",
        "entity_type": None,
        "lei": None,
    }
    text = _build_search_text(row)
    assert text == "Acme"  # only non-empty field


def test_compute_corpus_fingerprint_deterministic() -> None:
    """Same inputs always produce the same fingerprint."""
    from entity_data_lakehouse.search import _compute_corpus_fingerprint

    ids = ["E1", "E2"]
    texts = ["Acme Solar DE OPERATOR", "Nordic Wind NO OPERATOR"]
    fp1 = _compute_corpus_fingerprint(ids, texts, "all-MiniLM-L6-v2")
    fp2 = _compute_corpus_fingerprint(ids, texts, "all-MiniLM-L6-v2")
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex digest


def test_compute_corpus_fingerprint_text_change_detected() -> None:
    """Changing a search text field produces a different fingerprint."""
    from entity_data_lakehouse.search import _compute_corpus_fingerprint

    ids = ["E1"]
    fp_before = _compute_corpus_fingerprint(ids, ["Acme Solar DE"], "all-MiniLM-L6-v2")
    fp_after = _compute_corpus_fingerprint(ids, ["Acme Wind DE"], "all-MiniLM-L6-v2")
    assert fp_before != fp_after


def test_compute_corpus_fingerprint_model_change_detected() -> None:
    """Changing the model name produces a different fingerprint."""
    from entity_data_lakehouse.search import _compute_corpus_fingerprint

    ids = ["E1"]
    texts = ["Acme Solar DE"]
    fp_a = _compute_corpus_fingerprint(ids, texts, "all-MiniLM-L6-v2")
    fp_b = _compute_corpus_fingerprint(ids, texts, "all-mpnet-base-v2")
    assert fp_a != fp_b


def test_compute_corpus_fingerprint_order_independent() -> None:
    """Fingerprint is stable regardless of the input list order."""
    from entity_data_lakehouse.search import _compute_corpus_fingerprint

    ids_ab = ["E1", "E2"]
    texts_ab = ["text one", "text two"]
    ids_ba = ["E2", "E1"]
    texts_ba = ["text two", "text one"]
    assert _compute_corpus_fingerprint(ids_ab, texts_ab, "m") == \
        _compute_corpus_fingerprint(ids_ba, texts_ba, "m")


# ---------------------------------------------------------------------------
# Integration tests — require [search] extras.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold_duckdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full pipeline into a temp directory and return the DuckDB path."""
    import shutil

    repo_root = Path(__file__).resolve().parents[2]
    tmp_root = tmp_path_factory.mktemp("search_gold")

    shutil.copytree(repo_root / "sample_data", tmp_root / "sample_data")
    shutil.copytree(repo_root / "reference_data", tmp_root / "reference_data")
    shutil.copytree(repo_root / "contracts", tmp_root / "contracts")

    from entity_data_lakehouse.pipeline import run_pipeline

    run_pipeline(tmp_root)
    return tmp_root / "gold" / "entity_lakehouse.duckdb"


@pytest.fixture(scope="module")
def search_index(gold_duckdb: Path):  # type: ignore[return]
    """Build the hybrid search index once for the module.

    Qdrant is forced to in-memory mode so the fixture leaves no on-disk state.
    """
    from entity_data_lakehouse.search import build_search_index

    return build_search_index(gold_duckdb, qdrant_path=Path(":memory:"))


class TestBuildSearchIndex:
    @_requires_search
    def test_index_builds_without_error(self, search_index) -> None:  # type: ignore[no-untyped-def]
        assert search_index is not None

    @_requires_search
    def test_entity_rows_loaded(self, search_index) -> None:  # type: ignore[no-untyped-def]
        assert len(search_index._entity_rows) > 0

    @_requires_search
    def test_bm25_retriever_populated(self, search_index) -> None:  # type: ignore[no-untyped-def]
        """bm25s retriever is built and bm25_ids matches entity rows."""
        assert search_index._bm25_retriever is not None
        assert len(search_index._bm25_ids) == len(search_index._entity_rows)

    @_requires_search
    def test_qdrant_collection_populated(self, search_index) -> None:  # type: ignore[no-untyped-def]
        info = search_index._qdrant.get_collection("entity_master")
        assert info.points_count == len(search_index._entity_rows)

    @_requires_search
    def test_fingerprint_written_on_disk_build(
        self,
        gold_duckdb: Path,
        tmp_path: Path,
    ) -> None:
        """Fingerprint file is written after an on-disk build."""
        from entity_data_lakehouse.search import _FINGERPRINT_FILE, build_search_index

        qdrant_path = tmp_path / "qdrant_store"
        build_search_index(gold_duckdb, qdrant_path=qdrant_path)
        fp_file = qdrant_path / _FINGERPRINT_FILE
        assert fp_file.exists()
        assert len(fp_file.read_text().strip()) == 64  # SHA-256 hex

    @_requires_search
    def test_fingerprint_reuse_skips_rebuild(
        self,
        gold_duckdb: Path,
        tmp_path: Path,
    ) -> None:
        """Second call with unchanged corpus reuses collection (no re-embed)."""
        from entity_data_lakehouse.search import _FINGERPRINT_FILE, build_search_index

        qdrant_path = tmp_path / "qdrant_store"
        build_search_index(gold_duckdb, qdrant_path=qdrant_path)
        fp_after_first = (qdrant_path / _FINGERPRINT_FILE).read_text()
        build_search_index(gold_duckdb, qdrant_path=qdrant_path)
        fp_after_second = (qdrant_path / _FINGERPRINT_FILE).read_text()
        assert fp_after_first == fp_after_second


class TestHybridSearch:
    @_requires_search
    def test_returns_results(self, search_index) -> None:  # type: ignore[no-untyped-def]
        results = search_index.search("energy", top_k=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    @_requires_search
    def test_result_type(self, search_index) -> None:  # type: ignore[no-untyped-def]
        from entity_data_lakehouse.search import SearchResult

        results = search_index.search("solar", top_k=1)
        assert all(isinstance(r, SearchResult) for r in results)

    @_requires_search
    def test_top_k_respected(self, search_index) -> None:  # type: ignore[no-untyped-def]
        for k in (1, 2, 3):
            results = search_index.search("energy", top_k=k)
            assert len(results) <= k

    @_requires_search
    def test_rrf_scores_descending(self, search_index) -> None:  # type: ignore[no-untyped-def]
        results = search_index.search("infrastructure", top_k=5)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @_requires_search
    def test_rrf_scores_positive(self, search_index) -> None:  # type: ignore[no-untyped-def]
        results = search_index.search("operator", top_k=5)
        assert all(r.rrf_score > 0 for r in results)

    @_requires_search
    def test_result_has_required_fields(self, search_index) -> None:  # type: ignore[no-untyped-def]
        results = search_index.search("energy", top_k=2)
        for r in results:
            assert r.entity_id
            assert r.entity_name
            assert r.country_code
            assert r.entity_type

    @_requires_search
    def test_empty_query_still_returns(self, search_index) -> None:  # type: ignore[no-untyped-def]
        results = search_index.search("x", top_k=5)
        assert isinstance(results, list)

    @_requires_search
    def test_rank_provenance(self, search_index) -> None:  # type: ignore[no-untyped-def]
        """At least some results should have a vector_rank."""
        results = search_index.search("solar wind", top_k=5)
        vector_ranked = [r for r in results if r.vector_rank is not None]
        assert len(vector_ranked) >= 1
