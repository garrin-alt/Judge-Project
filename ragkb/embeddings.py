"""Local text embeddings via fastembed (no external API calls)."""

import threading

import numpy as np

from . import config

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding

                _model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning an (N, D) float32 array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_model()
    vectors = list(model.embed(texts))
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string, returning a (D,) float32 vector."""
    return embed_texts([text])[0]
