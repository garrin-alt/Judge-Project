"""Citation-graph expansion: follow rule references found in retrieved text.

Rules documents cite each other ("See rule 307.") and the community FAQ
cites official rules ("[ 354.2 ]"). After retrieval, we scan the hits for
such citations and pull in the cited rule documents as secondary context,
so an answer isn't limited to what similarity search happened to surface.

Rule documents are recognized by their titles: "Riftbound Core Rules §344"
or grouped "Riftbound Core Rules §350–352" (see ingest_rules_to_kb.py).
"""

import re
from dataclasses import dataclass

from .store import KnowledgeStore

# "See rule 307." / "see rule 307" / "in section 700"
_SEE_RULE_RE = re.compile(r"\b(?:[Ss]ee rules?|[Ss]ections?) (\d{3})")
# FAQ-style bracket citations: "[ 354.2 ]", "[383.3.c]"
_BRACKET_RE = re.compile(r"\[\s*(\d{3})(?:\.\d+[a-z0-9.]*)?\s*\]")
# rule-document titles: "... §344" or "... §350–352"
_TITLE_SPAN_RE = re.compile(r"^(.*Rules) §(\d{3})(?:–(\d{3}))?")


@dataclass
class RelatedRule:
    doc_id: int
    title: str
    source: str | None
    text: str
    cited_by: str


def extract_citations(text: str) -> list[str]:
    """Rule numbers cited in a piece of text, in order of appearance."""
    seen = []
    for m in _SEE_RULE_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    for m in _BRACKET_RE.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def build_rule_index(store: KnowledgeStore) -> dict[str, list[tuple[int, str]]]:
    """Map doc-name -> [(doc_id, span_start, span_end)] parsed from titles."""
    index: dict[str, list] = {}
    for doc_id, title in store.find_documents_by_title_prefix("Riftbound"):
        m = _TITLE_SPAN_RE.match(title)
        if not m:
            continue
        doc_name, lo, hi = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        index.setdefault(doc_name, []).append((doc_id, lo, hi))
    return index


def resolve_rule(index: dict, number: str, prefer_doc: str | None = None) -> int | None:
    """Find the document covering a rule number.

    prefer_doc: doc name of the citing chunk — its own numbering wins
    (Tournament Rules citations refer to tournament sections); otherwise
    Core Rules is the default namespace.
    """
    n = int(number)
    order = []
    if prefer_doc and prefer_doc in index:
        order.append(prefer_doc)
    for name in ("Riftbound Core Rules", "Riftbound Tournament Rules"):
        if name not in order and name in index:
            order.append(name)
    for name in order:
        for doc_id, lo, hi in index[name]:
            if lo <= n <= hi:
                return doc_id
    return None


def _citing_doc_name(title: str) -> str | None:
    m = _TITLE_SPAN_RE.match(title)
    return m.group(1) if m else None


def expand_keyword_rules(
    store: KnowledgeStore,
    results,
    glossary: list[dict],
    max_related: int = 2,
    exclude: set | None = None,
) -> list[RelatedRule]:
    """Link keywords in retrieved card text ([Assault], [Ambush]...) to the
    rules that define them, so a card result carries how it actually works."""
    if not results:
        return []
    keyword_rules = {
        e["term"]: e["rule"] for e in glossary
        if e.get("kind") == "keyword" and e.get("rule")
    }
    if not keyword_rules:
        return []
    index = build_rule_index(store)
    if not index:
        return []

    exclude = set(exclude or ())
    exclude.update(r.doc_id for r in results)
    related: list[RelatedRule] = []

    for r in results:
        for term, rule_number in keyword_rules.items():
            if len(related) >= max_related:
                return related
            if f"[{term}" not in r.text:
                continue
            doc_id = resolve_rule(index, rule_number)
            if doc_id is None or doc_id in exclude:
                continue
            exclude.add(doc_id)
            title, source, text = store.get_document_text(doc_id)
            related.append(RelatedRule(
                doc_id=doc_id, title=title, source=source, text=text,
                cited_by=r.title,
            ))
    return related


def expand_citations(store: KnowledgeStore, results, max_related: int = 3) -> list[RelatedRule]:
    """Follow citations in retrieved results to the cited rule documents."""
    if not results:
        return []
    index = build_rule_index(store)
    if not index:
        return []

    primary_ids = {r.doc_id for r in results}
    related: list[RelatedRule] = []
    seen_docs = set()

    for r in results:
        prefer = _citing_doc_name(r.title)
        for number in extract_citations(r.text):
            if len(related) >= max_related:
                return related
            doc_id = resolve_rule(index, number, prefer_doc=prefer)
            if doc_id is None or doc_id in primary_ids or doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            title, source, text = store.get_document_text(doc_id)
            related.append(RelatedRule(
                doc_id=doc_id, title=title, source=source, text=text,
                cited_by=r.title,
            ))
    return related
