import os
from pathlib import Path

DEFAULT_DB_PATH = Path(os.environ.get("RAGKB_DB_PATH", str(Path.home() / ".ragkb" / "kb.db")))
EMBEDDING_MODEL = os.environ.get("RAGKB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# "fastembed" (onnxruntime; desktop default) or "llama" (llama.cpp GGUF;
# works everywhere llama-cpp-python compiles, incl. Android/Termux).
EMBED_BACKEND = os.environ.get("RAGKB_EMBED_BACKEND", "fastembed")
EMBED_MODEL_URL = os.environ.get(
    "RAGKB_EMBED_MODEL_URL",
    "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/"
    "bge-small-en-v1.5-q8_0.gguf",
)
ANTHROPIC_MODEL = os.environ.get("RAGKB_ANTHROPIC_MODEL", "claude-sonnet-5")
CHUNK_SIZE = int(os.environ.get("RAGKB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("RAGKB_CHUNK_OVERLAP", "100"))
