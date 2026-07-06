import numpy as np
import pytest

from ragkb.store import KnowledgeStore


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "test.db")
    yield s
    s.close()


def _vec(*values):
    return np.array(values, dtype=np.float32)


def test_add_and_list_document(store):
    chunks = ["chunk one", "chunk two"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc_id = store.add_document("My Doc", "source.txt", chunks, vectors)
    assert doc_id == 1

    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0][1] == "My Doc"
    assert docs[0][4] == 2  # chunk count


def test_search_returns_closest_match(store):
    chunks = ["about cats", "about dogs", "about cars"]
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    store.add_document("Animals", None, chunks, vectors)

    results = store.search(_vec(0.0, 1.0, 0.0), top_k=1)
    assert len(results) == 1
    assert results[0].text == "about dogs"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_empty_store_returns_empty(store):
    assert store.search(_vec(1.0, 0.0), top_k=5) == []


def test_remove_document(store):
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    doc_id = store.add_document("Doc", None, ["text"], vectors)
    assert store.remove_document(doc_id) is True
    assert store.list_documents() == []
    assert store.remove_document(doc_id) is False


def test_reset_clears_everything(store):
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    store.add_document("Doc", None, ["text"], vectors)
    store.reset()
    assert store.count_chunks() == 0
    assert store.list_documents() == []


def test_add_document_mismatched_lengths_raises(store):
    with pytest.raises(ValueError):
        store.add_document("Doc", None, ["a", "b"], np.array([[1.0, 0.0]], dtype=np.float32))
