#!/usr/bin/env python3
"""Ingest the official Riftbound rules documents into a ragkb knowledge base.

Consumes data/riftbound_rules.json (from scripts/fetch_riftbound_rules.py):
Core Rules and Tournament Rules split per numbered rule, plus errata and
patch-notes article sections. Long sections are chunked. Re-runnable —
previously ingested official-rules entries are replaced, not duplicated.

Usage:
    python scripts/ingest_rules_to_kb.py [--rules data/riftbound_rules.json]
                                         [--db data/riftbound_kb.db]
"""

import argparse
import json
from pathlib import Path

from ragkb import config
from ragkb.chunking import chunk_text
from ragkb.embeddings import embed_texts
from ragkb.store import KnowledgeStore

# Sources whose documents this script owns (removed before re-ingesting).
OFFICIAL_PREFIXES = (
    "https://cmsassets.rgpub.io/",
    "https://riftbound.leagueoflegends.com/",
    "https://playriftbound.com/",
)


def section_to_chunks(section: dict) -> list[str]:
    header = f"{section['title']}\n"
    text = header + section["text"]
    if len(text) <= config.CHUNK_SIZE * 2:
        return [text]
    return [header + c for c in chunk_text(section["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="data/riftbound_rules.json")
    parser.add_argument("--db", default="data/riftbound_kb.db")
    args = parser.parse_args()

    sections = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    print(f"{len(sections)} rule sections")

    docs = [(s, section_to_chunks(s)) for s in sections]
    all_chunks = [c for _, chunks in docs for c in chunks]
    print(f"Embedding {len(all_chunks)} chunk(s)...")
    vectors = embed_texts(all_chunks)

    with KnowledgeStore(args.db) as store:
        stale = [
            d[0] for d in store.list_documents()
            if (d[2] or "").startswith(OFFICIAL_PREFIXES)
        ]
        for doc_id in stale:
            store.remove_document(doc_id)
        if stale:
            print(f"Removed {len(stale)} previously ingested official-rules document(s)")

        offset = 0
        for section, chunks in docs:
            vecs = vectors[offset:offset + len(chunks)]
            offset += len(chunks)
            store.add_document(
                title=section["title"],
                source=section["url"],
                chunks=chunks,
                embeddings=vecs,
            )
        total_docs = len(store.list_documents())
        total_chunks = store.count_chunks()

    print(f"Done. KB now has {total_docs} documents / {total_chunks} chunks at {args.db}")


if __name__ == "__main__":
    main()
