"""Find the specific things a question is about.

Judge questions are usually about named entities interacting: two cards, a
card and a keyword, a rule and a mechanic. Embedding the whole sentence
once blurs those together, so a question naming Card A and Card B can come
back with neither. This module pulls the named entities out so retrieval
can guarantee each one is represented.
"""

import re
from dataclasses import dataclass

from .store import KnowledgeStore

# Card names that are ordinary English words; matching these as entities
# produces false anchors ("can I flash a unit?" is not about the card
# Flash). Require extra evidence for short/common names.
_MIN_UNAMBIGUOUS_LEN = 8

_RULE_REF_RE = re.compile(r"(?:rule|section|§)\s*(\d{3})(?:\.\d+[a-z0-9.]*)?", re.IGNORECASE)
_CARD_CODE_RE = re.compile(r"\b([A-Za-z]{2,4}-\d{1,4}[A-Za-z]?)\b")


@dataclass(frozen=True)
class Entity:
    kind: str      # "card" | "rule" | "keyword"
    name: str      # canonical name as it appears in the knowledge base
    doc_id: int | None = None


def _card_index(store: KnowledgeStore) -> list[tuple[str, str, int]]:
    """[(lowercase name, canonical name, doc_id)], longest names first so
    "Vex, Apathetic" wins over "Vex"."""
    rows = store.conn.execute(
        "SELECT name, doc_id, id FROM cards WHERE doc_id IS NOT NULL"
    ).fetchall()
    seen: dict[str, tuple[str, int]] = {}
    for name, doc_id, _ in rows:
        key = name.lower()
        seen.setdefault(key, (name, doc_id))
    return sorted(
        ((k, v[0], v[1]) for k, v in seen.items()),
        key=lambda t: -len(t[0]),
    )


def extract_entities(prompt: str, store: KnowledgeStore, glossary: list[dict] | None = None) -> list[Entity]:
    """Named cards, rules and keywords mentioned in the prompt."""
    found: list[Entity] = []
    lower = prompt.lower()
    consumed: list[tuple[int, int]] = []  # spans already claimed by a longer name

    def overlaps(start: int, end: int) -> bool:
        return any(s < end and start < e for s, e in consumed)

    # exact card codes first — unambiguous
    for m in _CARD_CODE_RE.finditer(prompt):
        row = store.conn.execute(
            "SELECT name, doc_id FROM cards WHERE UPPER(id) = UPPER(?)", (m.group(1),)
        ).fetchone()
        if row and row[1]:
            found.append(Entity("card", row[0], row[1]))
            consumed.append(m.span())

    # card names, longest first
    for key, canonical, doc_id in _card_index(store):
        if len(key) < 4:
            continue
        for m in re.finditer(rf"(?<![\w-]){re.escape(key)}(?![\w-])", lower):
            if overlaps(*m.span()):
                continue
            ambiguous = len(key) < _MIN_UNAMBIGUOUS_LEN and " " not in key
            # a short, ordinary-looking name needs to be Capitalised in the
            # prompt to count as a card reference
            if ambiguous and not re.search(
                rf"(?<![\w-]){re.escape(canonical)}(?![\w-])", prompt
            ):
                continue
            if any(e.name == canonical for e in found):
                continue
            found.append(Entity("card", canonical, doc_id))
            consumed.append(m.span())
            break

    # explicit rule references
    for m in _RULE_REF_RE.finditer(prompt):
        name = f"§{m.group(1)}"
        if not any(e.name == name for e in found):
            found.append(Entity("rule", name))

    # glossary keywords (Assault, Ambush, ...) — bracketed or plain
    for entry in glossary or []:
        if entry.get("kind") != "keyword":
            continue
        term = entry["term"]
        if re.search(rf"(?<![\w-]){re.escape(term.lower())}(?![\w-])", lower):
            if not any(e.name == term for e in found):
                found.append(Entity("keyword", term))

    return found


def entity_doc_ids(store: KnowledgeStore, entities: list[Entity]) -> list[int]:
    """Documents that definitively cover the given entities."""
    ids: list[int] = []
    for e in entities:
        if e.doc_id is not None:
            ids.append(e.doc_id)
        elif e.kind == "rule":
            number = e.name.lstrip("§")
            row = store.conn.execute(
                "SELECT id FROM documents WHERE title LIKE ? LIMIT 1", (f"%§{number}%",)
            ).fetchone()
            if row:
                ids.append(row[0])
    return list(dict.fromkeys(ids))


# Clause splitting: judge questions often chain conditions.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and then|and|but|while|when|if|after|before|during|versus|vs\.?|against)\s+",
    re.IGNORECASE,
)


def subqueries(prompt: str, min_len: int = 12, max_parts: int = 3) -> list[str]:
    """Split a multi-part question into clauses worth retrieving separately.

    "Can I Ambush a unit while a showdown is open?" retrieves poorly as one
    vector — the two halves pull in different directions. Returns [] when
    the prompt is a single idea.
    """
    parts = [p.strip(" ?.,") for p in _CLAUSE_SPLIT_RE.split(prompt)]
    parts = [p for p in parts if len(p) >= min_len]
    if len(parts) < 2:
        return []
    return parts[:max_parts]
