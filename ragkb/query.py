"""The ragkb query pipeline: expand -> retrieve -> (generate).

All interfaces (CLI, REST API, web UI) run queries through here so
enhancements to querying apply everywhere at once.
"""

from dataclasses import dataclass, field

from .cardsearch import known_tags, parse_filters, search_cards
from .citations import RelatedRule, expand_citations
from .embeddings import embed_query
from .generate import GenerationUnavailable, generate_answer
from .glossary import expand_prompt
from .store import KnowledgeStore, SearchResult


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
) -> QueryResponse:
    """Run the full query pipeline against an open store.

    generate: "none" for retrieval only, else a backend name for
    generate_answer ("auto", "local", "claude").
    """
    resp = QueryResponse(prompt=prompt, expanded_prompt=prompt)

    if store.count_chunks() == 0:
        resp.notice = "Knowledge base is empty."
        return resp

    expansion = expand_prompt(prompt, store.get_glossary())
    resp.expanded_prompt = expansion.prompt
    resp.definitions = expansion.definitions

    qvec = embed_query(resp.expanded_prompt)

    glossary = store.get_glossary()
    filters = None
    if store.count_cards():
        filters = parse_filters(prompt, glossary, known_tags(store))

    if filters and filters.triggers_card_search(prompt):
        resp.card_filters = filters.describe()
        resp.results = search_cards(store, filters, qvec, top_k=top_k)
        if not resp.results:
            resp.notice = (
                f"No cards match the filters ({resp.card_filters}). "
                "Showing closest semantic matches instead."
            )
            resp.results = store.search(qvec, top_k=top_k)
    else:
        resp.results = store.search(qvec, top_k=top_k)

    resp.related = expand_citations(store, resp.results)

    if generate != "none" and resp.results:
        try:
            resp.answer, resp.backend = generate_answer(
                prompt, resp.results, backend=generate, model=model,
                definitions=resp.definitions, related=resp.related,
            )
        except GenerationUnavailable as e:
            resp.notice = str(e)

    return resp
