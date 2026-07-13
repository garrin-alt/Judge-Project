#!/usr/bin/env python3
"""Ingest the official Riftbound rules documents into a ragkb knowledge base.

Consumes data/riftbound_rules.json (from scripts/fetch_riftbound_rules.py)
and restructures the numbered rules hierarchically before embedding:

- Header rules ("341. Showdowns") are not stored as documents; they become
  breadcrumbs on the rules they enclose, so every chunk carries its topic:
  "Riftbound Core Rules §344 — Showdowns: 344. A Showdown begins when..."
- Small sections (a header plus its rules, e.g. a step-by-step procedure)
  are stored as ONE document so procedures stay together.
- Large sections stay per-rule (still with breadcrumbs); long rules are
  chunked.

Errata and patch-notes article sections are ingested as-is (chunked).
Re-runnable: previously ingested official-rules entries are replaced.

Usage:
    python scripts/ingest_rules_to_kb.py [--rules data/riftbound_rules.json]
                                         [--db data/riftbound_kb.db]
"""

import argparse
import json
import re
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

# One grouped section document may be at most this long; otherwise the
# section's rules are stored individually.
GROUP_MAX_CHARS = 3200


def is_header(section: dict) -> bool:
    """Header rules are short noun phrases with no sentence content.

    '341. Showdowns' -> header. '117. Players each draw 4.' -> content
    (it ends with a period: an actual instruction, not a title).
    """
    body = re.sub(rf"^{section['section']}\.\s*", "", section["text"]).strip()
    return len(body) <= 48 and not body.endswith(".") and f"{section['section']}.1" not in section["text"]


def split_subrules(rule_text: str, rule_number: str) -> list[str]:
    """Split a rule into its sub-rule units ("702.1.", "702.2.a.", ...).

    Only sub-numbers of THIS rule are boundaries, so citations to other
    rules ("See rule 307.") never cause a split. Text before the first
    sub-rule (the rule's own statement) is its own unit.
    """
    pat = re.compile(rf"(?:(?<=\s)|^)({re.escape(rule_number)}(?:\.\d+|\.[a-z]+)+\.)\s")
    matches = list(pat.finditer(rule_text))
    if not matches:
        return [rule_text]
    units = []
    head = rule_text[: matches[0].start()].strip()
    if head:
        units.append(head)
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(rule_text)
        unit = rule_text[m.start():end].strip()
        if unit:
            units.append(unit)
    return units


def pack_units(units: list[str], max_chars: int) -> list[str]:
    """Pack consecutive units into chunks of at most max_chars, cutting
    only between units. A single oversized unit falls back to
    sentence-aware splitting."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_text(unit, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
            continue
        if current and len(current) + 1 + len(unit) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def build_rule_docs(rules: list[dict]) -> list[dict]:
    """Group a document's rules into sections with breadcrumbs.

    Returns dicts: {title, text, url, units, sections: [rule numbers]}.
    Units are the atomic pieces chunking may cut between: whole rules for
    grouped sections, sub-rules for a single large rule.
    """
    doc_name = rules[0]["doc"]
    docs = []
    crumb: list[str] = []
    prev_was_header = False
    current: list[dict] = []

    def flush():
        nonlocal current
        if not current:
            return
        crumb_text = " › ".join(crumb) if crumb else None
        numbers = [r["section"] for r in current]
        combined = "\n".join(r["text"] for r in current)
        span = f"§{numbers[0]}" if len(numbers) == 1 else f"§{numbers[0]}–{numbers[-1]}"
        header_line = f"{crumb_text}\n" if crumb_text else ""
        suffix = f" — {crumb_text}" if crumb_text else ""

        if len(combined) <= GROUP_MAX_CHARS:
            docs.append({
                "title": f"{doc_name} {span}{suffix}",
                "text": header_line + combined,
                "url": current[0]["url"],
                "units": [r["text"] for r in current],
                "sections": numbers,
            })
        else:
            for r in current:
                docs.append({
                    "title": f"{doc_name} §{r['section']}{suffix}",
                    "text": header_line + r["text"],
                    "url": r["url"],
                    "units": split_subrules(r["text"], r["section"]),
                    "sections": [r["section"]],
                })
        current = []

    for rule in rules:
        if is_header(rule):
            flush()
            body = re.sub(rf"^{rule['section']}\.\s*", "", rule["text"]).strip()
            # consecutive headers nest ("Game Concepts" › "Deck Construction");
            # a header arriving after content starts a fresh breadcrumb
            crumb = crumb + [body] if prev_was_header else [body]
            prev_was_header = True
        else:
            prev_was_header = False
            current.append(rule)
    flush()
    return docs


def doc_to_chunks(doc: dict) -> list[str]:
    """Chunk a document, cutting only at unit (rule/sub-rule) boundaries;
    articles without units use sentence-aware splitting."""
    header = f"{doc['title']}\n"
    budget = config.MAX_CHUNK_CHARS - len(header)
    if len(doc["text"]) <= budget:
        return [header + doc["text"]]
    units = doc.get("units")
    if units:
        pieces = pack_units(units, budget)
    else:
        pieces = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return [header + p for p in pieces]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="data/riftbound_rules.json")
    parser.add_argument("--db", default="data/riftbound_kb.db")
    args = parser.parse_args()

    sections = json.loads(Path(args.rules).read_text(encoding="utf-8"))

    pdf_docs = []
    article_docs = []
    for doc_name in sorted({s["doc"] for s in sections}):
        doc_rules = [s for s in sections if s["doc"] == doc_name]
        if doc_name.endswith("Rules"):  # Core Rules / Tournament Rules PDFs
            pdf_docs.extend(build_rule_docs(doc_rules))
        else:  # errata / patch-notes articles
            for s in doc_rules:
                article_docs.append({
                    "title": s["title"],
                    "text": s["text"],
                    "url": s["url"],
                })

    all_docs = pdf_docs + article_docs
    print(f"{len(sections)} raw sections -> {len(all_docs)} documents "
          f"({len(pdf_docs)} rules, {len(article_docs)} article sections)")

    prepared = [(d, doc_to_chunks(d)) for d in all_docs]
    all_chunks = [c for _, chunks in prepared for c in chunks]
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
        for doc, chunks in prepared:
            vecs = vectors[offset:offset + len(chunks)]
            offset += len(chunks)
            store.add_document(
                title=doc["title"],
                source=doc["url"],
                chunks=chunks,
                embeddings=vecs,
            )
        total_docs = len(store.list_documents())
        total_chunks = store.count_chunks()

    print(f"Done. KB now has {total_docs} documents / {total_chunks} chunks at {args.db}")


if __name__ == "__main__":
    main()
