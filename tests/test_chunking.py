from ragkb.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_single_chunk():
    assert chunk_text("hello world", chunk_size=800, overlap=100) == ["hello world"]


def test_long_text_produces_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100
    # reconstructing should preserve all words in order (overlap aside)
    assert "word0" in chunks[0]
    assert "word499" in chunks[-1]


def test_invalid_overlap_raises():
    try:
        chunk_text("hello", chunk_size=10, overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_invalid_chunk_size_raises():
    try:
        chunk_text("hello", chunk_size=0, overlap=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
