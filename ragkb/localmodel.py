"""Fully offline answer generation via llama.cpp.

Downloads a small instruct model (GGUF) once, caches it under
~/.ragkb/models/, and runs CPU inference locally. After the first
download, generation works with no network access at all.
"""

import os
import sys
import threading
import urllib.request
from pathlib import Path

DEFAULT_MODEL_URL = os.environ.get(
    "RAGKB_LOCAL_MODEL_URL",
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-1.5b-instruct-q4_k_m.gguf",
)
MODEL_DIR = Path(os.environ.get("RAGKB_MODEL_DIR", str(Path.home() / ".ragkb" / "models")))

_llm = None
_llm_lock = threading.Lock()


def model_path() -> Path:
    return MODEL_DIR / DEFAULT_MODEL_URL.rsplit("/", 1)[-1]


def is_available() -> bool:
    """True if the local backend can run right now (model on disk + llama_cpp)."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return model_path().exists()


def ensure_model() -> Path:
    """Download the model file if it isn't cached yet."""
    path = model_path()
    if path.exists():
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")

    def report(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and sys.stderr.isatty():
            pct = min(100, done * 100 // total)
            print(f"\rDownloading local model: {pct}% of {total // 2**20} MiB", end="", file=sys.stderr)

    print(f"Downloading local model from {DEFAULT_MODEL_URL}", file=sys.stderr)
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, tmp, reporthook=report)
    print(file=sys.stderr)
    tmp.rename(path)
    return path


def _get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                from llama_cpp import Llama

                _llm = Llama(
                    model_path=str(ensure_model()),
                    n_ctx=4096,
                    verbose=False,
                )
    return _llm


def generate(system: str, user: str, max_tokens: int = 512) -> str:
    llm = _get_llm()
    with _llm_lock:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
    return result["choices"][0]["message"]["content"].strip()
