"""SQLite-backed storage for documents, chunks, and their embeddings."""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS glossary (
    term TEXT PRIMARY KEY,
    aliases TEXT NOT NULL,
    definition TEXT NOT NULL,
    rule TEXT,
    kind TEXT NOT NULL DEFAULT 'term'
);

CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    set_name TEXT,
    rarity TEXT,
    type TEXT,
    supertype TEXT,
    colors TEXT NOT NULL DEFAULT '[]',
    cost INTEGER,
    might INTEGER,
    tags TEXT NOT NULL DEFAULT '[]',
    keywords TEXT NOT NULL DEFAULT '[]',
    effect TEXT,
    url TEXT
);
"""


@dataclass
class SearchResult:
    chunk_id: int
    doc_id: int
    title: str
    source: str | None
    text: str
    score: float


class KnowledgeStore:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else config.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        glossary_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(glossary)")}
        if "kind" not in glossary_cols:
            self.conn.execute("ALTER TABLE glossary ADD COLUMN kind TEXT NOT NULL DEFAULT 'term'")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def add_document(self, title: str, source: str | None, chunks: list[str], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO documents (title, source, created_at) VALUES (?, ?, ?)",
            (title, source, time.time()),
        )
        doc_id = cur.lastrowid
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (doc_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                (doc_id, i, text, np.asarray(vec, dtype=np.float32).tobytes()),
            )
        self.conn.commit()
        return doc_id

    def list_documents(self):
        cur = self.conn.execute(
            "SELECT d.id, d.title, d.source, d.created_at, COUNT(c.id) "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY d.id ORDER BY d.created_at DESC"
        )
        return cur.fetchall()

    def remove_document(self, doc_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def reset(self):
        self.conn.executescript("DELETE FROM chunks; DELETE FROM documents;")
        self.conn.commit()

    def count_chunks(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def set_glossary(self, entries: list[dict]):
        """Replace the stored glossary. Entries: {term, aliases, definition, rule, kind}."""
        import json as _json

        self.conn.execute("DELETE FROM glossary")
        self.conn.executemany(
            "INSERT INTO glossary (term, aliases, definition, rule, kind) VALUES (?, ?, ?, ?, ?)",
            [
                (e["term"], _json.dumps(e.get("aliases", [])), e["definition"],
                 e.get("rule"), e.get("kind", "term"))
                for e in entries
            ],
        )
        self.conn.commit()

    def get_glossary(self) -> list[dict]:
        import json as _json

        rows = self.conn.execute(
            "SELECT term, aliases, definition, rule, kind FROM glossary"
        ).fetchall()
        return [
            {"term": r[0], "aliases": _json.loads(r[1]), "definition": r[2],
             "rule": r[3], "kind": r[4]}
            for r in rows
        ]

    def replace_cards(self, rows: list[dict]):
        """Replace the structured cards table. Each row mirrors the cards schema."""
        import json as _json

        self.conn.execute("DELETE FROM cards")
        self.conn.executemany(
            "INSERT INTO cards (id, doc_id, name, set_name, rarity, type, supertype,"
            " colors, cost, might, tags, keywords, effect, url)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r["id"], r.get("doc_id"), r["name"], r.get("set_name"), r.get("rarity"),
                 r.get("type"), r.get("supertype"),
                 _json.dumps(r.get("colors", [])), r.get("cost"), r.get("might"),
                 _json.dumps(r.get("tags", [])), _json.dumps(r.get("keywords", [])),
                 r.get("effect"), r.get("url"))
                for r in rows
            ],
        )
        self.conn.commit()

    def count_cards(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    def find_documents_by_title_prefix(self, prefix: str) -> list[tuple]:
        return self.conn.execute(
            "SELECT id, title FROM documents WHERE title LIKE ?", (prefix + "%",)
        ).fetchall()

    def get_document_text(
        self, doc_id: int, max_chars: int | None = 1200
    ) -> tuple[str, str | None, str]:
        """Return (title, source, text) for a document.

        Text is capped at max_chars; pass None for the full text. Chunks
        overlap (see chunking.py), so the concatenation may repeat a few
        words at chunk seams.
        """
        doc = self.conn.execute(
            "SELECT title, source FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if doc is None:
            raise KeyError(f"no document #{doc_id}")
        rows = self.conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        text = "\n".join(r[0] for r in rows)
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        return doc[0], doc[1], text

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        allowed_doc_ids: set[int] | None = None,
    ) -> list[SearchResult]:
        """Top matching chunks, at most one (the best) per document.

        allowed_doc_ids restricts the search to those documents (None = all).
        """
        rows = self.conn.execute(
            "SELECT c.id, c.doc_id, c.text, c.embedding, d.title, d.source "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id"
        ).fetchall()
        if allowed_doc_ids is not None:
            rows = [r for r in rows if r[1] in allowed_doc_ids]
        if not rows:
            return []

        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matrix = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-12
        scores = (matrix @ query_vec) / (norms * query_norm)

        results = []
        seen_docs = set()
        for idx in np.argsort(-scores):
            r = rows[idx]
            if r[1] in seen_docs:
                continue
            seen_docs.add(r[1])
            results.append(
                SearchResult(
                    chunk_id=r[0],
                    doc_id=r[1],
                    title=r[4],
                    source=r[5],
                    text=r[2],
                    score=float(scores[idx]),
                )
            )
            if len(results) >= top_k:
                break
        return results

    def count_doc_chunks(self, doc_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
