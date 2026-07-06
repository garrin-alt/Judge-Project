"""Generate a grounded answer from retrieved knowledge-base chunks using Claude."""

import os

from .store import SearchResult

SYSTEM_PROMPT = (
    "You are a knowledge-base assistant. Answer the user's question using ONLY the "
    "provided context excerpts. If the context does not contain the answer, say so "
    "plainly instead of guessing. Cite which source(s) you used by title."
)


class GenerationUnavailable(RuntimeError):
    """Raised when an answer can't be generated (e.g. missing API key)."""


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[Source: {r.title}]\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, results: list[SearchResult], model: str) -> str:
    if not results:
        raise GenerationUnavailable("No knowledge-base results to ground an answer on.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise GenerationUnavailable(
            "ANTHROPIC_API_KEY is not set; cannot generate an answer. "
            "Use --no-generate to see raw retrieved chunks instead."
        )

    import anthropic

    client = anthropic.Anthropic()
    context = build_context(results)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
