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
# bge models support an instruction prefix on the QUERY side (documents are
# embedded bare). Off by default: measured on this corpus it HURT — recall@1
# 93.8% -> 85.4%, MRR 0.965 -> 0.921 (scripts/eval_retrieval.py). The prefix
# suits prose passages; our documents are short structured records (card
# entries, numbered rule fragments), so it pulls queries away from them.
# Set RAGKB_QUERY_PREFIX to try it on a different corpus — and re-measure.
# Written out here rather than delegating to a backend helper so both
# embedding backends produce identical query vectors.
QUERY_PREFIX = os.environ.get("RAGKB_QUERY_PREFIX", "")
ANTHROPIC_MODEL = os.environ.get("RAGKB_ANTHROPIC_MODEL", "claude-sonnet-5")
CHUNK_SIZE = int(os.environ.get("RAGKB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("RAGKB_CHUNK_OVERLAP", "100"))
# Hard cap for any single stored chunk: the embedding model reads ~512
# tokens, so text beyond roughly this many characters would be stored but
# invisible to search.
MAX_CHUNK_CHARS = int(os.environ.get("RAGKB_MAX_CHUNK_CHARS", "1400"))

# Weight of the keyword (BM25) arm when fusing with dense retrieval.
# The keyword arm only fires on exact identifiers (card codes, dotted rule
# numbers) — see store._fts_query — where a match is near-certain, so it
# outranks a merely-similar dense hit. Measured with
# scripts/eval_retrieval.py: no change to the 48-question eval at any
# weight, and 2.0 is where exact card-code lookups ("OGN-224") land their
# card at rank 1. 0 disables keyword search entirely.
LEXICAL_WEIGHT = float(os.environ.get("RAGKB_LEXICAL_WEIGHT", "2.0"))
