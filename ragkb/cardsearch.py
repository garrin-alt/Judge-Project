"""Hybrid structured search for cards.

Questions like "fury cards with Assault that cost less than 1" are
database queries, not similarity searches. This module parses hard
constraints out of a prompt — domains/colors (via the glossary), numeric
cost/might comparisons, card types, keywords, tags — filters the
structured cards table with real logic, and ranks the survivors
semantically. If nothing matches, it says so instead of returning
look-alike cards.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np

from .store import KnowledgeStore, SearchResult

TYPE_WORDS = {
    "unit": "Unit", "units": "Unit",
    "spell": "Spell", "spells": "Spell",
    "gear": "Gear", "gears": "Gear",
    "rune": "Rune", "runes": "Rune",
    "battlefield": "Battlefield", "battlefields": "Battlefield",
    "legend": "Legend", "legends": "Legend",
}
SUPERTYPE_WORDS = {"champion": "Champion", "champions": "Champion"}

CARD_NOUNS = re.compile(r"\b(cards?|units?|spells?|gears?|champions?|legends?|battlefields?)\b", re.I)

_NUM_FIELDS = ("cost", "might")
_NUM_PATTERNS = [
    (r"{f}(?:s|ing)?\s+(?:of\s+)?(?:less than|under|below|fewer than)\s+(\d+)", "<"),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?(\d+)\s+or\s+(?:less|fewer|lower)", "<="),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?at most\s+(\d+)", "<="),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?(?:more than|over|above|greater than)\s+(\d+)", ">"),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?(\d+)\s+or\s+(?:more|higher|greater)", ">="),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?at least\s+(\d+)", ">="),
    (r"{f}(?:s|ing)?\s+(?:of\s+)?(?:exactly\s+)?(\d+)\b", "="),
]


@dataclass
class CardFilters:
    colors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)
    supertypes: list[str] = field(default_factory=list)
    cost: tuple[str, int] | None = None
    might: tuple[str, int] | None = None

    def describe(self) -> str:
        parts = []
        if self.colors:
            parts.append("domain " + "/".join(self.colors))
        if self.supertypes:
            parts.append("supertype " + "/".join(self.supertypes))
        if self.types:
            parts.append("type " + "/".join(self.types))
        if self.keywords:
            parts.append("keyword " + "/".join(self.keywords))
        if self.tags:
            parts.append("tag " + "/".join(self.tags))
        if self.cost:
            parts.append(f"cost {self.cost[0]} {self.cost[1]}")
        if self.might:
            parts.append(f"might {self.might[0]} {self.might[1]}")
        return ", ".join(parts)

    def triggers_card_search(self, prompt: str) -> bool:
        """Only route to structured search when the intent is clearly a
        card lookup: a hard constraint plus card-noun context, so rules
        questions that merely mention a keyword stay semantic."""
        hard = bool(self.colors or self.tags or self.cost or self.might)
        keyword_lookup = bool(self.keywords) and bool(re.search(r"\bcards?\b", prompt, re.I))
        return (hard or keyword_lookup) and bool(CARD_NOUNS.search(prompt))


def _word_hit(word: str, prompt: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", prompt, re.I) is not None


def parse_filters(prompt: str, glossary: list[dict], known_tags: list[str]) -> CardFilters:
    f = CardFilters()

    for entry in glossary:
        kind = entry.get("kind", "term")
        if kind == "domain":
            if _word_hit(entry["term"], prompt) or any(_word_hit(a, prompt) for a in entry.get("aliases", [])):
                f.colors.append(entry["term"])
        elif kind == "keyword" and _word_hit(entry["term"], prompt):
            f.keywords.append(entry["term"])

    for tag in known_tags:
        if len(tag) >= 3 and _word_hit(tag, prompt):
            f.tags.append(tag)

    lower = prompt.lower()
    for word, t in TYPE_WORDS.items():
        if re.search(rf"\b{word}\b", lower) and t not in f.types:
            f.types.append(t)
    for word, st in SUPERTYPE_WORDS.items():
        if re.search(rf"\b{word}\b", lower) and st not in f.supertypes:
            f.supertypes.append(st)

    for fname in _NUM_FIELDS:
        for pat, op in _NUM_PATTERNS:
            m = re.search(pat.format(f=fname), lower)
            if m:
                setattr(f, fname, (op, int(m.group(1))))
                break

    return f


def known_tags(store: KnowledgeStore) -> list[str]:
    tags = set()
    for (tags_json,) in store.conn.execute("SELECT tags FROM cards").fetchall():
        tags.update(json.loads(tags_json))
    return sorted(tags)


def _sql_for(filters: CardFilters) -> tuple[str, list]:
    where, params = [], []
    for color in filters.colors:
        where.append("colors LIKE ?")
        params.append(f'%"{color}"%')
    for kw in filters.keywords:
        where.append("keywords LIKE ?")
        params.append(f'%"{kw}"%')
    for tag in filters.tags:
        where.append("tags LIKE ?")
        params.append(f'%"{tag}"%')
    if filters.types:
        where.append("(" + " OR ".join("type = ?" for _ in filters.types) + ")")
        params.extend(filters.types)
    if filters.supertypes:
        where.append("(" + " OR ".join("supertype = ?" for _ in filters.supertypes) + ")")
        params.extend(filters.supertypes)
    for fname in _NUM_FIELDS:
        clause = getattr(filters, fname)
        if clause:
            op, val = clause
            op = {"=": "=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}[op]
            where.append(f"{fname} {op} ?")
            params.append(val)
    return (" AND ".join(where) or "1=1"), params


def search_cards(
    store: KnowledgeStore, filters: CardFilters, query_vec: np.ndarray, top_k: int = 5
) -> list[SearchResult]:
    """Filter the cards table, then rank the survivors by similarity."""
    where, params = _sql_for(filters)
    rows = store.conn.execute(
        f"SELECT c.doc_id, c.id, c.name, c.url FROM cards c WHERE {where}", params
    ).fetchall()
    if not rows:
        return []

    doc_ids = [r[0] for r in rows]
    placeholders = ",".join("?" for _ in doc_ids)
    chunk_rows = store.conn.execute(
        f"SELECT ch.doc_id, ch.text, ch.embedding, d.title, d.source "
        f"FROM chunks ch JOIN documents d ON d.id = ch.doc_id "
        f"WHERE ch.doc_id IN ({placeholders})",
        doc_ids,
    ).fetchall()
    if not chunk_rows:
        return []

    matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in chunk_rows])
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-12
    qnorm = np.linalg.norm(query_vec) or 1e-12
    scores = (matrix @ query_vec.astype(np.float32)) / (norms * qnorm)

    order = np.argsort(-scores)[:top_k]
    return [
        SearchResult(
            chunk_id=-1,
            doc_id=chunk_rows[i][0],
            title=chunk_rows[i][3],
            source=chunk_rows[i][4],
            text=chunk_rows[i][1],
            score=float(scores[i]),
        )
        for i in order
    ]
