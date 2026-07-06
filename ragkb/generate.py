"""Generate a grounded answer from retrieved knowledge-base chunks.

Two backends:
- "local":  fully offline via llama.cpp (see localmodel.py)
- "claude": Anthropic API (requires ANTHROPIC_API_KEY)
- "auto":   local if its model is cached, else Claude if a key is set
"""

import os

from . import config
from .store import SearchResult

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant. Answer the user's question using ONLY the "
    "provided context excerpts. If the context does not contain the answer, say so "
    "plainly instead of guessing. Cite which source(s) you used by title."
)


class GenerationUnavailable(RuntimeError):
    """Raised when an answer can't be generated (e.g. missing API key/model)."""


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[Source: {r.title}]\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def _build_user_message(
    query: str,
    results: list[SearchResult],
    definitions: list[str] | None = None,
    related=None,
    notes: list[str] | None = None,
) -> str:
    parts = []
    if notes:
        parts.append("Important notes:\n" + "\n".join(f"- {n}" for n in notes))
    if definitions:
        parts.append("Game term definitions:\n" + "\n".join(f"- {d}" for d in definitions))
    parts.append(f"Context:\n{build_context(results)}")
    if related:
        blocks = [f"[Referenced rule: {r.title}]\n{r.text}" for r in related]
        parts.append("Rules referenced by the context:\n" + "\n\n".join(blocks))
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)


def resolve_backend(backend: str = "auto") -> str:
    if backend in ("local", "claude"):
        return backend
    if backend != "auto":
        raise ValueError(f"unknown backend: {backend}")
    from . import localmodel

    if localmodel.is_available():
        return "local"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise GenerationUnavailable(
        "No generation backend available: the local model isn't downloaded "
        "(run `ragkb download-model`) and ANTHROPIC_API_KEY is not set."
    )


def generate_answer(
    query: str,
    results: list[SearchResult],
    backend: str = "auto",
    model: str | None = None,
    definitions: list[str] | None = None,
    related=None,
    notes: list[str] | None = None,
) -> tuple[str, str]:
    """Returns (answer, backend_used)."""
    if not results:
        raise GenerationUnavailable("No knowledge-base results to ground an answer on.")

    backend = resolve_backend(backend)
    user_message = _build_user_message(query, results, definitions, related, notes)

    if backend == "local":
        from . import localmodel

        return localmodel.generate(SYSTEM_PROMPT, user_message), "local"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise GenerationUnavailable(
            "ANTHROPIC_API_KEY is not set; cannot generate with the claude backend."
        )

    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, "claude"
