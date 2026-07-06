"""The ragkb query pipeline: expand -> retrieve -> (generate).

All interfaces (CLI, REST API, web UI) run queries through here so
enhancements to querying apply everywhere at once.
"""

from dataclasses import dataclass, field

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
    resp.results = store.search(qvec, top_k=top_k)

    if generate != "none" and resp.results:
        try:
            resp.answer, resp.backend = generate_answer(
                prompt, resp.results, backend=generate, model=model,
                definitions=resp.definitions,
            )
        except GenerationUnavailable as e:
            resp.notice = str(e)

    return resp
