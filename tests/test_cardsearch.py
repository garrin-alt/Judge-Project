import numpy as np
import pytest

from ragkb.cardsearch import parse_filters, search_cards
from ragkb.store import KnowledgeStore

GLOSSARY = [
    {"term": "Fury", "aliases": ["red"], "definition": "d", "rule": "134", "kind": "domain"},
    {"term": "Chaos", "aliases": ["purple"], "definition": "d", "rule": "134", "kind": "domain"},
    {"term": "Assault", "aliases": [], "definition": "d", "rule": "807", "kind": "keyword"},
    {"term": "Heal", "aliases": [], "definition": "d", "rule": "418", "kind": "term"},
]
TAGS = ["Yordle", "Noxus"]


def _vec(x=1.0):
    v = np.zeros(4, dtype=np.float32)
    v[0] = x
    return v


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(tmp_path / "kb.db")
    cards = [
        {"id": "A-1", "name": "Cheap Fury", "type": "Spell", "colors": ["Fury"],
         "cost": 1, "keywords": ["Assault"], "tags": []},
        {"id": "A-2", "name": "Big Fury", "type": "Unit", "supertype": "Champion",
         "colors": ["Fury"], "cost": 6, "might": 6, "keywords": ["Assault"], "tags": ["Noxus"]},
        {"id": "B-1", "name": "Purple Champ", "type": "Unit", "supertype": "Champion",
         "colors": ["Chaos"], "cost": 3, "might": 3, "keywords": [], "tags": []},
    ]
    rows = []
    for c in cards:
        doc_id = s.add_document(f"{c['name']} ({c['id']})", None,
                                [f"{c['name']} effect text"], _vec().reshape(1, -1))
        rows.append({**c, "doc_id": doc_id})
    s.replace_cards(rows)
    yield s
    s.close()


def test_parse_color_via_alias_and_keyword_and_cost():
    f = parse_filters("What fury cards have Assault that cost less than 1?", GLOSSARY, TAGS)
    assert f.colors == ["Fury"]
    assert f.keywords == ["Assault"]
    assert f.cost == ("<", 1)
    assert f.triggers_card_search("What fury cards have Assault that cost less than 1?")


def test_parse_supertype_and_alias_color():
    prompt = "What are my options for Purple chosen champion?"
    f = parse_filters(prompt, GLOSSARY, TAGS)
    assert f.colors == ["Chaos"]
    assert f.supertypes == ["Champion"]
    assert f.triggers_card_search(prompt)


def test_rules_question_does_not_trigger():
    prompt = "when do triggered abilities trigger?"
    f = parse_filters(prompt, GLOSSARY, TAGS)
    assert not f.triggers_card_search(prompt)


def test_keyword_mention_without_card_context_does_not_trigger():
    prompt = "who gets to act after I Ambush a unit during a showdown?"
    f = parse_filters(prompt, GLOSSARY, TAGS)
    assert not f.triggers_card_search(prompt)


def test_search_applies_all_filters(store):
    f = parse_filters("fury cards with Assault costing less than 2", GLOSSARY, TAGS)
    results = search_cards(store, f, _vec())
    assert [r.title for r in results] == ["Cheap Fury (A-1)"]


def test_search_honest_empty(store):
    f = parse_filters("fury cards with Assault that cost less than 1", GLOSSARY, TAGS)
    assert search_cards(store, f, _vec()) == []


def test_search_champion_by_color(store):
    f = parse_filters("purple champions", GLOSSARY, TAGS)
    results = search_cards(store, f, _vec())
    assert [r.title for r in results] == ["Purple Champ (B-1)"]


def test_cost_or_less_and_might(store):
    f = parse_filters("units with might 6 or more and cost 6 or less", GLOSSARY, TAGS)
    assert f.might == (">=", 6)
    assert f.cost == ("<=", 6)
    results = search_cards(store, f, _vec())
    assert [r.title for r in results] == ["Big Fury (A-2)"]
