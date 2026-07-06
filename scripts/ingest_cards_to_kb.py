#!/usr/bin/env python3
"""Ingest the collated Riftbound card dataset into a ragkb knowledge base.

Each card becomes one document whose text combines the fields that matter
for retrieval (name, type, colors, tags, rules text, flavor). Promo and
showcase reprints with identical name + rules text are collapsed so query
results aren't dominated by duplicates.

Usage:
    python scripts/ingest_cards_to_kb.py [--cards data/riftbound_cards.json]
                                         [--db data/riftbound_kb.db]
"""

import argparse
import json
import re
from pathlib import Path

from ragkb.embeddings import embed_texts
from ragkb.store import KnowledgeStore

KEYWORD_RE = re.compile(r"\[([A-Z][a-zA-Z ]+?)(?:\s+\d+)?\]")


def card_to_row(card: dict, doc_id: int) -> dict:
    """Structured row for the cards table (typed columns for filtering)."""
    def as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return {
        "id": card["id"],
        "doc_id": doc_id,
        "name": card["name"],
        "set_name": card.get("set_name"),
        "rarity": card.get("rarity"),
        "type": card.get("type") or None,
        "supertype": card.get("supertype") or None,
        "colors": card.get("color") or [],
        "cost": as_int(card.get("cost")),
        "might": as_int(card.get("might")),
        "tags": card.get("tags") or [],
        "keywords": sorted({m.strip() for m in KEYWORD_RE.findall(card.get("effect") or "")}),
        "effect": card.get("effect"),
        "url": card.get("url"),
    }


def card_to_text(card: dict) -> str:
    parts = [f"{card['name']} ({card['id']})"]

    facts = []
    kind = " ".join(filter(None, [card.get("supertype"), card.get("type")]))
    if kind:
        facts.append(kind)
    if card.get("color"):
        facts.append("Color: " + ", ".join(card["color"]))
    if card.get("cost") is not None:
        facts.append(f"Cost {card['cost']}")
    if card.get("might") is not None:
        facts.append(f"Might {card['might']}")
    facts.append(f"{card['rarity']}, {card['set_name']}")
    parts.append(". ".join(facts) + ".")

    if card.get("tags"):
        parts.append("Tags: " + ", ".join(card["tags"]) + ".")
    if card.get("effect"):
        parts.append("Effect: " + card["effect"])
    # The feed stores current text in `errata` even when nothing changed;
    # only keep it when it actually differs from the effect text.
    if card.get("errata") and card["errata"].strip() != (card.get("effect") or "").strip():
        parts.append("Errata: " + card["errata"])
    if card.get("flavor"):
        parts.append("Flavor: " + card["flavor"])
    return "\n".join(parts)


def dedupe(cards: list[dict]) -> list[dict]:
    """Collapse reprints: same name + rules text keeps the first (base) copy,
    preferring non-promo printings."""
    cards = sorted(cards, key=lambda c: (c["promo"], c["id"]))
    seen = {}
    for card in cards:
        key = (card["name"], card["effect"])
        if key not in seen:
            seen[key] = card
    return sorted(seen.values(), key=lambda c: c["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", default="data/riftbound_cards.json")
    parser.add_argument("--db", default="data/riftbound_kb.db")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    cards = json.loads(Path(args.cards).read_text(encoding="utf-8"))
    unique = dedupe(cards)
    print(f"{len(cards)} cards -> {len(unique)} after collapsing reprints")

    texts = [card_to_text(c) for c in unique]
    print(f"Embedding {len(texts)} cards...")
    vectors = embed_texts(texts)

    with KnowledgeStore(args.db) as store:
        # Re-running should refresh, not duplicate: drop previous card docs.
        stale = [d[0] for d in store.list_documents()
                 if (d[2] or "").startswith("https://riftbound.gg")]
        for doc_id in stale:
            store.remove_document(doc_id)
        if stale:
            print(f"Removed {len(stale)} previously ingested card document(s)")

        rows = []
        for card, text, vec in zip(unique, texts, vectors):
            doc_id = store.add_document(
                title=f"{card['name']} ({card['id']})",
                source=card["url"],
                chunks=[text],
                embeddings=vec.reshape(1, -1),
            )
            rows.append(card_to_row(card, doc_id))
        store.replace_cards(rows)
        total = store.count_cards()
        print(f"Structured cards table: {total} rows")

    print(f"Done. {total} unique cards in knowledge base at {args.db}")
    print(f'Try: ragkb --db {args.db} query "which cards counter spells?"')


if __name__ == "__main__":
    main()
