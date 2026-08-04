"""The ragkb query pipeline: expand -> retrieve -> (generate).

All interfaces (CLI, REST API, web UI) run queries through here so
enhancements to querying apply everywhere at once.
"""

from dataclasses import dataclass, field

from . import config
from .cardsearch import known_tags, parse_filters, search_cards
from .citations import RelatedRule, expand_citations, expand_keyword_rules
from .embeddings import embed_query
from .entities import Entity, entity_doc_ids, extract_entities, subqueries
from .generate import GenerationUnavailable, generate_answer
from .glossary import expand_prompt
from .store import KnowledgeStore, SearchResult


SOURCE_KINDS = ("card", "faq", "core", "tournament", "errata", "patch", "note")


def doc_kind(title: str | None, source: str | None) -> str:
    """Classify a document by origin (mirrors the web UI's badge logic).

    Order matters: patch-notes titles also start with "Riftbound Core
    Rules", and errata titles mention cards.
    """
    t, s = title or "", source or ""
    if "Errata" in t or "errata" in s:
        return "errata"
    if "Patch Notes" in t or "patch-notes" in s:
        return "patch"
    if t.startswith("Riftbound Core Rules"):
        return "core"
    if t.startswith("Riftbound Tournament Rules"):
        return "tournament"
    if t.startswith("FAQ:") or "riftboundfaq" in s:
        return "faq"
    if "riftbound.gg/cards" in s:
        return "card"
    return "note"


def _allowed_doc_ids(store: KnowledgeStore, sources: list[str]) -> set[int]:
    wanted = set(sources)
    rows = store.conn.execute("SELECT id, title, source FROM documents").fetchall()
    return {r[0] for r in rows if doc_kind(r[1], r[2]) in wanted}


@dataclass
class QueryResponse:
    prompt: str
    expanded_prompt: str
    definitions: list[str] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)
    related: list[RelatedRule] = field(default_factory=list)
    card_filters: str | None = None
    entities: list[Entity] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    reranked: bool = False
    answer: str | None = None
    backend: str | None = None
    notice: str | None = None


def _entity_aware_search(
    store: KnowledgeStore,
    prompt: str,
    expanded: str,
    qvec,
    top_k: int,
    allowed: set[int] | None,
    glossary: list[dict],
    resp: "QueryResponse",
) -> list[SearchResult]:
    """Retrieval that keeps a question's separate ideas separate.

    Three passes, merged:
    1. the whole question (as before);
    2. each clause of a multi-part question, embedded on its own — the two
       halves of "can I do X while Y is happening" pull in different
       directions when averaged into one vector;
    3. the documents for entities named in the question (cards, rules,
       keywords), which are guaranteed a slot so a question about two
       named cards can never come back with neither.
    """
    base = store.search(qvec, top_k=top_k, allowed_doc_ids=allowed, keyword_query=prompt)

    resp.entities = extract_entities(prompt, store, glossary)
    resp.subqueries = subqueries(expanded)

    if not resp.entities and not resp.subqueries:
        return base

    merged: list[SearchResult] = []
    seen_docs: set[int] = set()

    def take(results, limit=None):
        added = 0
        for r in results:
            if r.doc_id in seen_docs:
                continue
            if allowed is not None and r.doc_id not in allowed:
                continue
            seen_docs.add(r.doc_id)
            merged.append(r)
            added += 1
            if limit and added >= limit:
                return

    # the single best whole-question hit leads: when a question has a
    # direct answer (a FAQ ruling, say), naming a card in it must not
    # demote that answer below the card's own text
    take(base, limit=1)

    # then the named entities — guaranteed a slot, not guaranteed the top
    anchor_ids = entity_doc_ids(store, resp.entities)
    if anchor_ids:
        anchors = store.search(qvec, top_k=len(anchor_ids), allowed_doc_ids=set(anchor_ids))
        take(anchors)

    # then one hit per clause, so each idea is represented
    for clause in resp.subqueries:
        if len(merged) >= top_k:
            break
        clause_hits = store.search(embed_query(clause), top_k=2, allowed_doc_ids=allowed)
        take(clause_hits, limit=1)

    # top up from the base ranking if there is room left
    take(base)
    return merged[:top_k]


def run_query(
    store: KnowledgeStore,
    prompt: str,
    top_k: int = 5,
    generate: str = "none",
    model: str | None = None,
    sources: list[str] | None = None,
) -> QueryResponse:
    """Run the full query pipeline against an open store.

    generate: "none" for retrieval only, else a backend name for
    generate_answer ("auto", "local", "claude").
    sources: restrict results to these document kinds (see SOURCE_KINDS);
    None or empty means all. Referenced-rule expansion is not restricted —
    a filtered result may still cite rules outside the filter.
    """
    resp = QueryResponse(prompt=prompt, expanded_prompt=prompt)

    if store.count_chunks() == 0:
        resp.notice = "Knowledge base is empty."
        return resp

    expansion = expand_prompt(prompt, store.get_glossary())
    resp.expanded_prompt = expansion.prompt
    resp.definitions = expansion.definitions

    qvec = embed_query(resp.expanded_prompt)
    store.check_embedding_provenance(int(qvec.shape[-1]))

    allowed = _allowed_doc_ids(store, sources) if sources else None

    glossary = store.get_glossary()
    filters = None
    cards_allowed = sources is None or not sources or "card" in sources
    if store.count_cards() and cards_allowed:
        filters = parse_filters(prompt, glossary, known_tags(store))

    generation_notes = []
    if filters and filters.triggers_card_search(prompt):
        resp.card_filters = filters.describe()
        resp.results = search_cards(store, filters, qvec, top_k=top_k)
        if not resp.results:
            resp.notice = (
                f"No cards match the filters ({resp.card_filters}). "
                "Showing closest semantic matches instead."
            )
            resp.results = store.search(qvec, top_k=top_k, allowed_doc_ids=allowed,
                                        keyword_query=prompt)
            generation_notes.append(
                f"A database search found NO cards matching all of: {resp.card_filters}. "
                "The context below contains only near-matches. State clearly that no "
                "card satisfies the question exactly, then describe the closest options."
            )
    else:
        # Two-stage: gather a wide shortlist, then let the cross-encoder
        # decide the order of what's actually shown.
        from . import rerank as _rerank

        wide = max(top_k, config.RERANK_CANDIDATES) if _rerank.is_available() else top_k
        shortlist = _entity_aware_search(
            store, prompt, resp.expanded_prompt, qvec, wide, allowed, glossary, resp
        )
        resp.results = _rerank.rerank(prompt, shortlist, top_k)
        resp.reranked = len(shortlist) > len(resp.results) and _rerank.is_available()

    resp.related = expand_citations(store, resp.results)
    resp.related += expand_keyword_rules(
        store, resp.results, glossary,
        exclude={r.doc_id for r in resp.related},
    )

    if generate != "none" and resp.results:
        try:
            resp.answer, resp.backend = generate_answer(
                prompt, resp.results, backend=generate, model=model,
                definitions=resp.definitions, related=resp.related,
                notes=generation_notes,
            )
        except GenerationUnavailable as e:
            resp.notice = str(e)

    return resp
