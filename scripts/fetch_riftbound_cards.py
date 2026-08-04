#!/usr/bin/env python3
"""Fetch and collate the Riftbound card database used by riftbound.gg.

riftbound.gg is part of the dotgg.gg network; its card browser loads data
from the dotgg card API. This script downloads that dataset and collates
the gameplay-relevant fields (card text, tags, cost, etc.) into JSON and
CSV, dropping imagery, art, and market-price fields.

Usage:
    python scripts/fetch_riftbound_cards.py [--out-dir data]
"""

import argparse
import csv
import html
import json
import re
import urllib.request
from pathlib import Path

API_URL = "https://api.dotgg.gg/cgfw/getcards?game=riftbound&mode=indexed"
CARD_URL_PREFIX = "https://riftbound.gg/cards/"

# Gameplay/text fields to keep, in output order. Everything else in the
# feed (image URLs, market ids, prices) is intentionally dropped.
KEEP_FIELDS = [
    "id",
    "name",
    "set_name",
    "rarity",
    "type",
    "supertype",
    "color",
    "cost",
    "might",
    "tags",
    "effect",
    "flavor",
    "promo",
    "banned",
    "errata",
]

TAG_RE = re.compile(r"<[^>]+>")

# The feed marks game icons with :rb_*: tokens; render them as readable text.
ICON_RE = re.compile(r":rb_([a-z_0-9]+):")


def _icon_to_text(match: re.Match) -> str:
    token = match.group(1)
    if token.startswith("energy_"):
        return f"{{{token.removeprefix('energy_')} Energy}}"
    if token == "rune_rainbow":
        return "{Any Rune}"
    if token.startswith("rune_"):
        return f"{{{token.removeprefix('rune_').title()} Rune}}"
    return "{" + token.replace("_", " ").title() + "}"


def clean_text(value):
    """Convert the API's HTML-ish rich text to plain text."""
    if not isinstance(value, str):
        return value
    value = re.sub(r"<br\s*/?>", "\n", value)
    value = TAG_RE.sub("", value)
    value = ICON_RE.sub(_icon_to_text, value)
    return html.unescape(value).strip()


def fetch(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Referer": "https://riftbound.gg/",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def collate(raw: dict) -> list[dict]:
    idx = {name: i for i, name in enumerate(raw["names"])}
    cards = []
    for row in raw["data"]:
        card = {}
        for field in KEEP_FIELDS:
            value = row[idx[field]]
            if field in ("effect", "flavor", "errata"):
                value = clean_text(value)
            elif field in ("promo", "banned"):
                value = value in ("1", 1, True)
            card[field] = value
        card["url"] = CARD_URL_PREFIX + row[idx["slug"]] + "/"
        cards.append(card)
    cards.sort(key=lambda c: c["id"])
    return cards


def write_json(cards: list[dict], path: Path):
    path.write_text(json.dumps(cards, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(cards: list[dict], path: Path):
    fields = KEEP_FIELDS + ["url"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            row = dict(card)
            for key in ("color", "tags"):
                row[key] = "; ".join(row[key] or [])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {API_URL} ...")
    raw = fetch(API_URL)
    cards = collate(raw)

    json_path = out_dir / "riftbound_cards.json"
    csv_path = out_dir / "riftbound_cards.csv"
    write_json(cards, json_path)
    write_csv(cards, csv_path)

    sets = {}
    for c in cards:
        sets[c["set_name"]] = sets.get(c["set_name"], 0) + 1
    print(f"Collated {len(cards)} cards -> {json_path} and {csv_path}")
    for name, count in sorted(sets.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
