import numpy as np
import pytest

from ragkb.citations import expand_citations, extract_citations
from ragkb.store import KnowledgeStore


def _vec():
    return np.ones((1, 4), dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "kb.db")
    s.add_document("Riftbound Core Rules §307 — States", None,
                   ["307. States of the Turn. A state is..."], _vec())
    s.add_document("Riftbound Core Rules §350–352 — Playing Cards", None,
                   ["350. Playing a card...\n352. Card states..."], _vec())
    s.add_document("Riftbound Tournament Rules §700 — Penalties", None,
                   ["700. Penalty guidelines..."], _vec())
    yield s
    s.close()


def test_extract_citations_variants():
    assert extract_citations("See rule 307. States for more.") == ["307"]
    assert extract_citations("pauses [ 354.2 ] and [383.3.c] resolve") == ["354", "383"]
    assert extract_citations("no citations here [Assault 2]") == []


def test_expand_follows_citation(store):
    results = store.search(np.ones(4, dtype=np.float32), top_k=1)
    # make the retrieved chunk cite rule 350
    results[0].text = "This ability pauses. See rule 350. Playing for details."
    results[0].title = "FAQ: Something"
    related = expand_citations(store, results)
    assert len(related) == 1
    assert "§350–352" in related[0].title
    assert related[0].cited_by == "FAQ: Something"


def test_expand_skips_already_retrieved(store):
    results = store.search(np.ones(4, dtype=np.float32), top_k=3)
    for r in results:
        r.text = "See rule 307."
    related = expand_citations(store, results)
    # §307 doc is among primary results, so no expansion
    assert related == []


def test_tournament_numbering_preferred_for_tournament_chunks(store):
    results = store.search(np.ones(4, dtype=np.float32), top_k=1)
    results[0].title = "Riftbound Tournament Rules §205 — Conduct"
    results[0].text = "covered below in section 700."
    related = expand_citations(store, results)
    assert related and related[0].title.startswith("Riftbound Tournament Rules §700")
