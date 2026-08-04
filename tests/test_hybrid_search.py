import numpy as np
import pytest

from ragkb.store import KnowledgeStore, _fts_query


def _vec(*vals):
    return np.array([list(vals)], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "kb.db")
    # doc A is semantically closest to the query vector below;
    # doc B is the one whose text carries the exact card code.
    s.add_document("Near Miss (OGN-273)", None,
                   ["Near Miss (OGN-273) some other card"], _vec(1.0, 0.0))
    s.add_document("Salvage (OGN-224)", None,
                   ["Salvage (OGN-224) You may kill up to one gear. Draw 1."], _vec(0.0, 1.0))
    yield s
    s.close()


def test_fts_query_extracts_only_exact_identifiers():
    assert _fts_query("what does salvage do?") == ""
    assert _fts_query("OGN-224") == '"OGN-224"'
    assert _fts_query("rule 702.12.b.2 please") == '"702.12.b.2"'
    assert _fts_query("UNL-145A vs VEN-030") == '"UNL-145A" OR "VEN-030"'


def test_keyword_search_finds_card_code(store):
    if not store.fts_enabled:
        pytest.skip("SQLite build lacks FTS5")
    hits = store.keyword_chunks("OGN-224")
    assert hits, "expected a keyword hit for the exact card code"
    titles = {store.conn.execute("SELECT title FROM documents WHERE id=?", (d,)).fetchone()[0]
              for _, d, _ in hits}
    assert "Salvage (OGN-224)" in titles


def test_hybrid_lifts_exact_match_over_semantic_neighbour(store):
    if not store.fts_enabled:
        pytest.skip("SQLite build lacks FTS5")
    query = np.array([1.0, 0.0], dtype=np.float32)  # points at the WRONG card
    dense_only = store.search(query, top_k=1, keyword_query=None)
    assert dense_only[0].title == "Near Miss (OGN-273)"

    hybrid = store.search(query, top_k=1, keyword_query="OGN-224", lexical_weight=2.0)
    assert hybrid[0].title == "Salvage (OGN-224)"


def test_natural_language_query_does_not_engage_keyword_arm(store):
    if not store.fts_enabled:
        pytest.skip("SQLite build lacks FTS5")
    query = np.array([1.0, 0.0], dtype=np.float32)
    plain = store.search(query, top_k=2, keyword_query="what does salvage do?")
    dense = store.search(query, top_k=2)
    assert [r.title for r in plain] == [r.title for r in dense]


def test_fts_index_tracks_inserts_and_deletes(store):
    if not store.fts_enabled:
        pytest.skip("SQLite build lacks FTS5")
    doc_id = store.add_document("New (VEN-999)", None, ["New (VEN-999) text"], _vec(0.5, 0.5))
    assert store.keyword_chunks("VEN-999")
    store.remove_document(doc_id)
    assert not store.keyword_chunks("VEN-999")


def test_embedding_provenance_stamps_then_warns(tmp_path, monkeypatch):
    from ragkb import config

    s = KnowledgeStore(tmp_path / "prov.db")
    s.add_document("Doc", None, ["text"], _vec(1.0, 0.0))
    assert s.get_meta("embed:dim") == "2"
    assert s.get_meta("embed:backend") == config.EMBED_BACKEND

    # a different embedding width must not pass silently
    with pytest.warns(UserWarning, match="Embedding mismatch"):
        s.add_document("Doc2", None, ["text"], np.ones((1, 3), dtype=np.float32))
    s.close()


def test_search_hydrates_only_returned_rows(store):
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].text  # text is loaded for what's returned
    assert results[0].chunk_id > 0
