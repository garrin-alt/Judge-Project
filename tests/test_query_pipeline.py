import numpy as np
import pytest

import ragkb.query as query_mod
from ragkb.query import run_query
from ragkb.store import KnowledgeStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = KnowledgeStore(tmp_path / "kb.db")

    def fake_embed(text):
        import hashlib

        vec = np.zeros(8, dtype=np.float32)
        for token in text.lower().split():
            token = token.strip("()?,.")
            digest = hashlib.md5(token.encode()).digest()
            vec[digest[0] % 8] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    monkeypatch.setattr(query_mod, "embed_query", fake_embed)

    s.add_document("Chaos card", None, ["a Chaos champion unit"],
                   fake_embed("a Chaos champion unit").reshape(1, -1))
    s.add_document("Fury card", None, ["a Fury attacker"],
                   fake_embed("a Fury attacker").reshape(1, -1))
    s.set_glossary([
        {"term": "Chaos", "aliases": ["purple"], "definition": "Domain colored purple.", "rule": "134"},
    ])
    yield s
    s.close()


def test_expansion_changes_retrieval(store):
    resp = run_query(store, "purple champion", top_k=1)
    assert resp.expanded_prompt == "purple (Chaos) champion"
    assert resp.definitions and "purple" in resp.definitions[0]
    assert resp.results[0].title == "Chaos card"


def test_no_glossary_hit_passes_through(store):
    resp = run_query(store, "Fury attacker", top_k=1)
    assert resp.expanded_prompt == "Fury attacker"
    assert resp.results[0].title == "Fury card"


def test_empty_store_notice(tmp_path):
    with KnowledgeStore(tmp_path / "empty.db") as s:
        resp = run_query(s, "anything")
    assert resp.notice == "Knowledge base is empty."
    assert resp.results == []
