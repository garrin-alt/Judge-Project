"""Cross-encoder reranking: the second stage of two-stage retrieval.

Embedding search compares a query vector to a chunk vector — the two were
encoded independently, so nothing ever looks at the question and the
passage *together*. A cross-encoder does exactly that, scoring (query,
chunk) pairs jointly, which is why the standard production pattern is
"retrieve wide with embeddings, then rerank the shortlist".

Optional by design: if no reranker backend is installed the pipeline
falls back to first-stage ordering, so Android/Termux installs that only
have llama.cpp keep working.
"""

import threading

from . import config
from .store import SearchResult

_model = None
_model_lock = threading.Lock()
_unavailable = False


def is_available() -> bool:
    if config.RERANK_BACKEND == "off":
        return False
    if _unavailable:
        return False
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    global _model, _unavailable
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _model = TextCrossEncoder(model_name=config.RERANK_MODEL)
    return _model


def rerank(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Re-order a shortlist by cross-encoder relevance.

    Returns the input unchanged (trimmed to top_k) when no reranker is
    available or anything goes wrong — reranking is an improvement, never
    a dependency.
    """
    global _unavailable
    if len(results) <= 1 or not is_available():
        return results[:top_k]
    try:
        model = _get_model()
        with _model_lock:
            scores = list(model.rerank(query, [r.text for r in results]))
    except Exception:
        _unavailable = True  # don't retry a broken backend on every query
        return results[:top_k]

    order = sorted(range(len(results)), key=lambda i: -scores[i])
    reranked = []
    for i in order[:top_k]:
        r = results[i]
        reranked.append(SearchResult(
            chunk_id=r.chunk_id, doc_id=r.doc_id, title=r.title,
            source=r.source, text=r.text, score=float(scores[i]),
        ))
    return reranked
