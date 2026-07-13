"""Local text embeddings (no external API calls).

Two interchangeable backends, selected with RAGKB_EMBED_BACKEND:
- "fastembed" (default): ONNX runtime, fastest on desktop.
- "llama": the same bge-small model as GGUF via llama.cpp — no
  onnxruntime dependency, so it works on Android/Termux and anywhere
  else llama-cpp-python compiles.

Embeddings stored in a knowledge base must come from the same backend
that embeds queries against it — pick one backend per KB and rebuild the
KB if you switch.
"""

import threading

import numpy as np

from . import config

_model = None
_model_lock = threading.Lock()


def _get_fastembed():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.EMBEDDING_MODEL)


def _get_llama_embedder():
    from llama_cpp import Llama

    from .localmodel import download_model_file

    path = download_model_file(config.EMBED_MODEL_URL)
    return Llama(model_path=str(path), embedding=True, n_ctx=512, verbose=False)


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if config.EMBED_BACKEND == "llama":
                    _model = _get_llama_embedder()
                else:
                    _model = _get_fastembed()
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returning an (N, D) float32 array of
    unit-length vectors."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _get_model()
    if config.EMBED_BACKEND == "llama":
        with _model_lock:
            vectors = [model.embed(t) for t in texts]
        arr = np.array(vectors, dtype=np.float32)
    else:
        arr = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return arr / norms


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string, returning a (D,) float32 vector."""
    return embed_texts([text])[0]
