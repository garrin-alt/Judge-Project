import numpy as np
import pytest
from fastapi.testclient import TestClient

import ragkb.server as server_mod
from ragkb.server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Deterministic fake embeddings so tests don't need the real model.
    def fake_embed_texts(texts):
        import hashlib

        rng = np.random.default_rng(0)
        out = []
        for t in texts:
            vec = np.zeros(8, dtype=np.float32)
            vec[hashlib.md5(t.encode()).digest()[0] % 8] = 1.0
            vec += rng.normal(scale=0.01, size=8).astype(np.float32)
            out.append(vec)
        return np.array(out, dtype=np.float32)

    monkeypatch.setattr(server_mod, "embed_texts", fake_embed_texts)
    monkeypatch.setattr("ragkb.query.embed_query", lambda q: fake_embed_texts([q])[0])

    app = create_app(db_path=str(tmp_path / "kb.db"))
    return TestClient(app)


def test_status_empty(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == 0
    assert body["chunks"] == 0
    assert set(body["backends"]) == {"local", "claude"}


def test_add_list_remove_document(client):
    r = client.post("/api/documents", json={"title": "Note", "text": "standup is at 9:30"})
    assert r.status_code == 200
    doc_id = r.json()["id"]

    docs = client.get("/api/documents").json()
    assert len(docs) == 1
    assert docs[0]["title"] == "Note"

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert client.get("/api/documents").json() == []

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 404


def test_add_empty_text_rejected(client):
    r = client.post("/api/documents", json={"title": "x", "text": "   "})
    assert r.status_code == 400


def test_query_retrieval(client):
    client.post("/api/documents", json={"title": "A", "text": "standup is at 9:30"})
    client.post("/api/documents", json={"title": "B", "text": "deploy happens on fridays"})

    r = client.post("/api/query", json={"prompt": "standup is at 9:30", "top_k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] is None
    assert len(body["results"]) == 1
    assert body["results"][0]["title"] == "A"


def test_query_empty_kb(client):
    r = client.post("/api/query", json={"prompt": "anything"})
    assert r.status_code == 200
    assert r.json()["notice"] == "Knowledge base is empty."


def test_query_generate_unavailable_falls_back(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("ragkb.localmodel.is_available", lambda: False)
    client.post("/api/documents", json={"title": "A", "text": "standup is at 9:30"})

    r = client.post("/api/query", json={"prompt": "standup", "generate": "auto"})
    body = r.json()
    assert body["answer"] is None
    assert "No generation backend available" in body["notice"]
    assert body["results"]  # retrieval still returned


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ragkb" in r.text
