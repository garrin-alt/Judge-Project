import os
from pathlib import Path

DEFAULT_DB_PATH = Path(os.environ.get("RAGKB_DB_PATH", str(Path.home() / ".ragkb" / "kb.db")))
EMBEDDING_MODEL = os.environ.get("RAGKB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
ANTHROPIC_MODEL = os.environ.get("RAGKB_ANTHROPIC_MODEL", "claude-sonnet-5")
CHUNK_SIZE = int(os.environ.get("RAGKB_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("RAGKB_CHUNK_OVERLAP", "100"))
