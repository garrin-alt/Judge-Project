"""The ragkb query pipeline: expand -> retrieve -> (generate).

All interfaces (CLI, REST API, web UI) run queries through here so
enhancements to querying apply everywhere at once.
"""

from dataclasses import dataclass, field

from .cardsearch import known_tags, parse_filters, search_cards
from .citations import RelatedRule, expand_citations, expand_keyword_rules
from .embeddings import embed_query
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
    answer: str | None = None
    backend: str | None = None
    notice: str | None = None


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
            resp.results = store.search(qvec, top_k=top_k, allowed_doc_ids=allowed)
            generation_notes.append(
                f"A database search found NO cards matching all of: {resp.card_filters}. "
                "The context below contains only near-matches. State clearly that no "
                "card satisfies the question exactly, then describe the closest options."
            )
    else:
        resp.results = store.search(qvec, top_k=top_k, allowed_doc_ids=allowed)

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
