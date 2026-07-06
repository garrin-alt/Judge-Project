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
from pathlib import Path

from ragkb.embeddings import embed_texts
from ragkb.store import KnowledgeStore


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
    if card.get("errata"):
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
        for card, text, vec in zip(unique, texts, vectors):
            store.add_document(
                title=f"{card['name']} ({card['id']})",
                source=card["url"],
                chunks=[text],
                embeddings=vec.reshape(1, -1),
            )
        total = store.count_chunks()

    print(f"Done. {total} cards in knowledge base at {args.db}")
    print(f'Try: ragkb --db {args.db} query "which cards counter spells?"')


if __name__ == "__main__":
    main()
