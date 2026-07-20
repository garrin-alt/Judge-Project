#!/usr/bin/env python3
"""Ingest the collated riftboundfaq.com Q&A sections into a ragkb knowledge base.

Each Q&A section becomes one document; long answers are chunked. Run after
scripts/ingest_cards_to_kb.py to combine rules FAQ and card data in one KB.

Usage:
    python scripts/ingest_faq_to_kb.py [--faq data/riftbound_faq.json]
                                       [--db data/riftbound_kb.db]
"""

import argparse
import json
from pathlib import Path

from ragkb import config
from ragkb.chunking import chunk_text
from ragkb.embeddings import embed_texts
from ragkb.store import KnowledgeStore


def _file_fingerprint(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def section_to_chunks(section: dict) -> list[str]:
    text = (
        f"Riftbound rules FAQ — {section['page']}\n"
        f"Q: {section['question']}\n"
        f"A: {section['answer']}"
    )
    if len(text) <= config.MAX_CHUNK_CHARS:
        return [text]
    header = f"Riftbound rules FAQ — {section['page']}\nQ: {section['question']}\n"
    return [header + c for c in chunk_text(section["answer"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faq", default="data/riftbound_faq.json")
    parser.add_argument("--db", default="data/riftbound_kb.db")
    args = parser.parse_args()

    sections = json.loads(Path(args.faq).read_text(encoding="utf-8"))
    print(f"{len(sections)} FAQ sections")

    docs = [(s, section_to_chunks(s)) for s in sections]
    all_chunks = [c for _, chunks in docs for c in chunks]
    print(f"Embedding {len(all_chunks)} chunk(s)...")
    vectors = embed_texts(all_chunks)

    with KnowledgeStore(args.db) as store:
        # Re-running should refresh, not duplicate: drop previous FAQ docs.
        stale = [d[0] for d in store.list_documents() if (d[2] or "").startswith("https://www.riftboundfaq.com")]
        for doc_id in stale:
            store.remove_document(doc_id)
        if stale:
            print(f"Removed {len(stale)} previously ingested FAQ document(s)")

        offset = 0
        for section, chunks in docs:
            vecs = vectors[offset:offset + len(chunks)]
            offset += len(chunks)
            store.add_document(
                title=f"FAQ: {section['page']} — {section['question']}",
                source=section["url"],
                chunks=chunks,
                embeddings=vecs,
            )
        store.set_meta("fingerprint:faq", _file_fingerprint(args.faq))
        total_docs = len(store.list_documents())
        total_chunks = store.count_chunks()

    print(f"Done. KB now has {total_docs} documents / {total_chunks} chunks at {args.db}")


if __name__ == "__main__":
    main()
