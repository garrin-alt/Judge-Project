from ragkb.glossary import expand_prompt

GLOSSARY = [
    {"term": "Chaos", "aliases": ["purple"], "definition": "Chaos is a Domain (color purple).", "rule": "134"},
    {"term": "Fury", "aliases": ["red"], "definition": "Fury is a Domain (color red).", "rule": "134"},
    {"term": "Assault", "aliases": [], "definition": "Assault N: +N Might while attacking.", "rule": "807"},
]


def test_alias_annotated_with_canonical_term():
    exp = expand_prompt("show me purple champions", GLOSSARY)
    assert exp.prompt == "show me purple (Chaos) champions"
    assert [m["term"] for m in exp.matches] == ["Chaos"]
    assert "Chaos" in exp.definitions[0]


def test_direct_term_matches_without_rewrite():
    exp = expand_prompt("how does Assault work?", GLOSSARY)
    assert exp.prompt == "how does Assault work?"
    assert [m["term"] for m in exp.matches] == ["Assault"]


def test_case_insensitive_alias():
    exp = expand_prompt("Purple cards", GLOSSARY)
    assert exp.prompt == "Purple (Chaos) cards"


def test_no_match_leaves_prompt_untouched():
    exp = expand_prompt("when does damage heal?", GLOSSARY)
    assert exp.prompt == "when does damage heal?"
    assert exp.matches == []


def test_no_partial_word_matches():
    # "reduce" must not match alias "red"
    exp = expand_prompt("reduce the cost", GLOSSARY)
    assert exp.matches == []


def test_alias_not_double_annotated_when_term_present():
    exp = expand_prompt("is purple the same as Chaos?", GLOSSARY)
    # canonical term already present; no rewrite needed
    assert "(Chaos)" not in exp.prompt
    assert [m["term"] for m in exp.matches] == ["Chaos"]
