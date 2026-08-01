"""SQLite-backed storage for documents, chunks, and their embeddings."""

import re
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

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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


# Words that carry no retrieval signal but match nearly every chunk, so
# ORing them into a BM25 query only adds noise.
_STOPWORDS = frozenset("""
a an and are as at be by can could do does doing for from had has have how i
if in into is it its me my of on or should so than that the their them then
there these they this to two up was what when where which who why will with
would you your
""".split())


# Exact identifiers embeddings are bad at: card codes (VEN-030, UNL-145A)
# and dotted rule numbers (354.2, 702.12.b.2).
_EXACT_TOKEN_RE = re.compile(r"\b(?:[A-Za-z]{2,4}-\d{1,4}[A-Za-z]?|\d{3}\.\d+(?:\.[A-Za-z0-9]+)*)\b")


def _fts_query(text: str) -> str:
    """Build an FTS5 query for the exact identifiers in a prompt.

    Only identifiers — measured on this corpus, ORing ordinary content
    words into BM25 and fusing it with dense retrieval made results
    strictly worse at every weight (recall@1 93.8% -> 89.6% at the mildest
    setting; see scripts/eval_retrieval.py). Dense retrieval already
    handles natural-language phrasing well; what it cannot do is match a
    card code or sub-rule number, so that is all the keyword arm is for.

    Terms are quoted so punctuation FTS5 treats as syntax can't produce a
    malformed expression.
    """
    terms = list(dict.fromkeys(_EXACT_TOKEN_RE.findall(text)))
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:8])


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
        self.fts_enabled: bool | None = None
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        glossary_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(glossary)")}
        if "kind" not in glossary_cols:
            self.conn.execute("ALTER TABLE glossary ADD COLUMN kind TEXT NOT NULL DEFAULT 'term'")
        self._ensure_fts()

    def _ensure_fts(self):
        """Create the FTS5 keyword index if this SQLite build supports it.

        Keyword search complements the embeddings on exact tokens the
        vectors handle poorly: rule numbers ("354.2"), bracketed keywords
        ("[Assault]"), and card names. Optional by design — SQLite builds
        without FTS5 keep working, dense-only.
        """
        if self.fts_enabled is not None:
            return
        try:
            fresh = not self.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            ).fetchone()
            self.conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, content='chunks', content_rowid='id',
                    tokenize='porter unicode61');
                CREATE TRIGGER IF NOT EXISTS chunks_fts_ins AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_fts_del AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_fts_upd AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END;
            """)
            self.fts_enabled = True
            # Existing knowledge bases predate the index: populate it once.
            if fresh and self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            self.fts_enabled = False

    def rebuild_fts(self):
        """Resynchronise the keyword index with the chunks table."""
        if not self.fts_enabled:
            return
        self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def check_embedding_provenance(self, dim: int, raise_on_mismatch: bool = False):
        """Record which embedder wrote this KB; complain if it changes.

        Vectors from different models/backends aren't comparable, so mixing
        them silently produces nonsense results. The first write stamps the
        KB; later ones are verified against that stamp.
        """
        from . import config

        current = {
            "embed:backend": config.EMBED_BACKEND,
            "embed:model": (config.EMBED_MODEL_URL.rsplit("/", 1)[-1]
                            if config.EMBED_BACKEND == "llama" else config.EMBEDDING_MODEL),
            "embed:dim": str(dim),
        }
        stored = {k: self.get_meta(k) for k in current}
        if all(v is None for v in stored.values()):
            for k, v in current.items():
                self.set_meta(k, v)
            return

        mismatches = [f"{k.split(':')[1]}: KB has {stored[k]!r}, this run uses {v!r}"
                      for k, v in current.items() if stored[k] is not None and stored[k] != v]
        if not mismatches:
            return
        message = (
            "Embedding mismatch — this knowledge base was built with a different "
            "embedder, so search results would be meaningless:\n  "
            + "\n  ".join(mismatches)
            + "\nRebuild the knowledge base with the current settings, or switch back."
        )
        if raise_on_mismatch:
            raise ValueError(message)
        import warnings

        warnings.warn(message, stacklevel=2)

    def add_document(self, title: str, source: str | None, chunks: list[str], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if len(embeddings):
            self.check_embedding_provenance(int(np.asarray(embeddings).shape[-1]))
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

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

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

    def score_chunks(
        self, query_vec: np.ndarray, allowed_doc_ids: set[int] | None = None
    ) -> list[tuple[int, int, float]]:
        """Cosine-score every chunk. Returns [(chunk_id, doc_id, score)].

        Only ids and vectors are read here — chunk text is fetched later
        for the handful of rows that actually get returned.
        """
        rows = self.conn.execute("SELECT id, doc_id, embedding FROM chunks").fetchall()
        if allowed_doc_ids is not None:
            rows = [r for r in rows if r[1] in allowed_doc_ids]
        if not rows:
            return []

        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-12
        scores = (matrix @ query_vec) / (norms * query_norm)
        return [(rows[i][0], rows[i][1], float(scores[i])) for i in range(len(rows))]

    def hydrate(self, scored: list[tuple[int, int, float]], top_k: int) -> list[SearchResult]:
        """Take scored chunks (best first), keep one per document, and load
        their text/title/source."""
        picked, seen_docs = [], set()
        for chunk_id, doc_id, score in scored:
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            picked.append((chunk_id, doc_id, score))
            if len(picked) >= top_k:
                break
        if not picked:
            return []

        placeholders = ",".join("?" for _ in picked)
        rows = self.conn.execute(
            f"SELECT c.id, c.text, d.title, d.source FROM chunks c "
            f"JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({placeholders})",
            [p[0] for p in picked],
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        results = []
        for chunk_id, doc_id, score in picked:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            results.append(SearchResult(
                chunk_id=chunk_id, doc_id=doc_id, title=row[2],
                source=row[3], text=row[1], score=score,
            ))
        return results

    def keyword_chunks(
        self, query: str, limit: int = 50, allowed_doc_ids: set[int] | None = None
    ) -> list[tuple[int, int, float]]:
        """BM25 keyword search over chunk text. [(chunk_id, doc_id, score)],
        best first. Empty if FTS5 is unavailable or the query has no usable
        terms."""
        if not self.fts_enabled:
            return []
        match = _fts_query(query)
        if not match:
            return []
        try:
            rows = self.conn.execute(
                "SELECT f.rowid, c.doc_id, bm25(chunks_fts) FROM chunks_fts f "
                "JOIN chunks c ON c.id = f.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed MATCH expression: fall back to dense only
        # bm25() returns negative numbers, lower = better; flip for sanity
        return [(r[0], r[1], -float(r[2])) for r in rows
                if allowed_doc_ids is None or r[1] in allowed_doc_ids]

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        allowed_doc_ids: set[int] | None = None,
        keyword_query: str | None = None,
        rrf_k: int = 60,
        lexical_weight: float | None = None,
    ) -> list[SearchResult]:
        """Top matching chunks, at most one (the best) per document.

        allowed_doc_ids restricts the search to those documents (None = all).
        keyword_query enables hybrid retrieval: dense and BM25 rankings are
        combined with Reciprocal Rank Fusion, which needs no score
        calibration between the two very different scales.
        """
        dense = self.score_chunks(query_vec, allowed_doc_ids)
        dense.sort(key=lambda t: -t[2])

        from . import config

        weight = config.LEXICAL_WEIGHT if lexical_weight is None else lexical_weight
        lexical = (self.keyword_chunks(keyword_query, allowed_doc_ids=allowed_doc_ids)
                   if keyword_query and weight > 0 else [])
        if not lexical:
            return self.hydrate(dense, top_k)

        fused: dict[int, list] = {}
        for rank, (chunk_id, doc_id, _) in enumerate(dense[: 200], start=1):
            fused[chunk_id] = [doc_id, 1.0 / (rrf_k + rank)]
        for rank, (chunk_id, doc_id, _) in enumerate(lexical, start=1):
            entry = fused.setdefault(chunk_id, [doc_id, 0.0])
            entry[1] += weight / (rrf_k + rank)

        combined = sorted(
            ((cid, v[0], v[1]) for cid, v in fused.items()), key=lambda t: -t[2]
        )
        return self.hydrate(combined, top_k)

    def count_doc_chunks(self, doc_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0]
