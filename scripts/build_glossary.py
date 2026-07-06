#!/usr/bin/env python3
"""Build a game-term glossary from the parsed Core Rules document.

Extracts, automatically:
- the six Domains with their colors as aliases (rule 134: "Chaos is
  associated with the color purple"), so user color words map to domains
- keyword definitions (Assault, Ambush, ...) for every keyword that
  appears in card text, located by their defining rule
- definition-style rules ("418. Heal ...", "131. Cost ...") as general
  game terms

Writes data/riftbound_glossary.json and, with --db, installs the glossary
into the knowledge base so the query pipeline uses it.

Usage:
    python scripts/build_glossary.py [--rules data/riftbound_rules.json]
                                     [--cards data/riftbound_cards.json]
                                     [--out data/riftbound_glossary.json]
                                     [--db data/riftbound_kb.db]
"""

import argparse
import json
import re
from pathlib import Path

DEF_SNIPPET_LEN = 300

# Words that qualify as terms but would false-positive constantly in
# ordinary questions ("card", "cards", "turn", "play"...).
STOP_TERMS = {
    "card", "cards", "player", "players", "turn", "turns", "game", "play",
    "playing cards", "name", "back side", "front side", "the board",
}


def _definition_snippet(text: str, term: str) -> str:
    body = re.sub(rf"^\d{{3}}\.\s+{re.escape(term)}\s*", "", text).strip()
    body = re.sub(r"\d{3}\.\d+[a-z.\d]*\.\s*", "", body)  # drop sub-rule numbers
    if len(body) > DEF_SNIPPET_LEN:
        body = body[:DEF_SNIPPET_LEN].rsplit(" ", 1)[0] + "..."
    return body


def extract_domains(core: list[dict]) -> list[dict]:
    entries = []
    rule134 = next((s for s in core if s["section"] == "134"), None)
    if not rule134:
        return entries
    for m in re.finditer(
        r"(\w+) is associated with the color (\w+) and represented by ([^.]+)\.", rule134["text"]
    ):
        domain, color, symbol = m.group(1), m.group(2), m.group(3)
        entries.append({
            "term": domain,
            "aliases": [color],
            "definition": f"{domain} is one of the six Domains, associated with the color {color} and {symbol.strip()}.",
            "rule": "134",
            "kind": "domain",
        })
    return entries


def extract_keywords(core: list[dict], cards: list[dict]) -> list[dict]:
    keywords = set()
    for c in cards:
        for kw in re.findall(r"\[([A-Z][a-zA-Z ]+?)(?:\s+\d+)?\]", c.get("effect") or ""):
            keywords.add(kw.strip())

    entries = []
    for kw in sorted(keywords):
        rule = next(
            (s for s in core if re.match(rf"^\d{{3}}\.\s+{re.escape(kw)}\b", s["text"])),
            None,
        )
        if rule is None:
            continue
        entries.append({
            "term": kw,
            "aliases": [],
            "definition": _definition_snippet(rule["text"], kw),
            "rule": rule["section"],
            "kind": "keyword",
        })
    return entries


def extract_definition_rules(core: list[dict]) -> list[dict]:
    """Rules shaped like '418. Heal 418.1. ...' define a named game term."""
    entries = []
    for s in core:
        m = re.match(rf"^{s['section']}\.\s+([A-Z][a-zA-Z' -]{{2,28}}?)\s+{s['section']}\.1", s["text"])
        if not m:
            continue
        term = m.group(1).strip()
        if term.lower() in STOP_TERMS or len(term.split()) > 3:
            continue
        entries.append({
            "term": term,
            "aliases": [],
            "definition": _definition_snippet(s["text"], term),
            "rule": s["section"],
            "kind": "term",
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="data/riftbound_rules.json")
    parser.add_argument("--cards", default="data/riftbound_cards.json")
    parser.add_argument("--out", default="data/riftbound_glossary.json")
    parser.add_argument("--db", default=None, help="Also install into this knowledge base.")
    args = parser.parse_args()

    secs = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    core = [s for s in secs if s["doc"] == "Riftbound Core Rules"]
    cards = json.loads(Path(args.cards).read_text(encoding="utf-8"))

    by_term: dict[str, dict] = {}
    # order matters: domains and keywords win over generic definition rules
    for entry in extract_definition_rules(core) + extract_keywords(core, cards) + extract_domains(core):
        by_term[entry["term"].lower()] = entry
    entries = sorted(by_term.values(), key=lambda e: e["term"])

    Path(args.out).write_text(
        json.dumps(entries, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(entries)} glossary entries to {args.out}")
    domains = [e for e in entries if e.get("kind") == "domain"]
    print("Domains:", ", ".join(f"{e['term']}({e['aliases'][0]})" for e in domains))

    if args.db:
        from ragkb.store import KnowledgeStore

        with KnowledgeStore(args.db) as store:
            store.set_glossary(entries)
        print(f"Installed glossary into {args.db}")


if __name__ == "__main__":
    main()
