#!/usr/bin/env python3
"""Check whether the knowledge base is up to date, at all three layers:

1. KB vs local data files — has data/ changed since the last ingest?
   (Uses fingerprints the ingest scripts store in the KB.)
2. Repo vs GitHub — are there unpulled commits, and do they touch data/?
3. Live sources vs repo — lightweight online checks against riftbound.gg,
   riftboundfaq.com, and the playriftbound.com rules hub (no PDF
   downloads: the hub's PDF links are content-hashed, so a changed link
   means a changed document).

Prints exactly which commands to run, if any. Use --offline to skip
layer 3.

Usage:
    python scripts/check_updates.py [--db data/riftbound_kb.db] [--offline]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from ragkb.store import KnowledgeStore

OK, WARN, ACT = "  ✓", "  ?", "  ✗"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

INGEST_COMMANDS = [
    "python scripts/ingest_cards_to_kb.py",
    "python scripts/ingest_faq_to_kb.py",
    "python scripts/ingest_rules_to_kb.py",
    "python scripts/build_glossary.py --db data/riftbound_kb.db",
]
FETCH_COMMANDS = [
    "python scripts/fetch_riftbound_cards.py",
    "python scripts/fetch_riftbound_faq.py",
    "python scripts/fetch_riftbound_rules.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def check_kb_vs_data(db: str) -> list[str]:
    print("\n[1] Knowledge base vs local data files")
    actions = []
    pairs = [
        ("data/riftbound_cards.json", "fingerprint:cards", "ingest_cards"),
        ("data/riftbound_faq.json", "fingerprint:faq", "ingest_faq"),
        ("data/riftbound_rules.json", "fingerprint:rules", "ingest_rules"),
    ]
    if not Path(db).exists():
        print(f"{ACT} no knowledge base at {db} — run all ingest scripts")
        return INGEST_COMMANDS

    with KnowledgeStore(db) as store:
        for file, key, _ in pairs:
            path = Path(file)
            if not path.exists():
                print(f"{ACT} {file} missing — run the fetch scripts")
                actions.extend(FETCH_COMMANDS)
                continue
            stored = store.get_meta(key)
            if stored is None:
                print(f"{WARN} {file}: KB predates fingerprint tracking — re-ingest once to enable")
                actions.extend(INGEST_COMMANDS)
            elif stored != sha256(path):
                print(f"{ACT} {file} changed since last ingest")
                actions.extend(INGEST_COMMANDS)
            else:
                print(f"{OK} {file} matches the KB")
    return list(dict.fromkeys(actions))


def check_repo_vs_remote() -> list[str]:
    print("\n[2] Repo vs GitHub")
    try:
        subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{WARN} not a git checkout — skipped")
        return []
    fetch = subprocess.run(["git", "fetch", "--quiet"], capture_output=True, text=True)
    if fetch.returncode != 0:
        print(f"{WARN} git fetch failed (offline?): {fetch.stderr.strip().splitlines()[-1] if fetch.stderr else 'unknown'}")
        return []
    behind = subprocess.run(
        ["git", "rev-list", "--count", "HEAD..@{upstream}"], capture_output=True, text=True
    )
    if behind.returncode != 0 or not behind.stdout.strip().isdigit():
        print(f"{WARN} no upstream tracking branch — skipped")
        return []
    n = int(behind.stdout.strip())
    if n == 0:
        print(f"{OK} up to date with GitHub")
        return []
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD...@{upstream}"], capture_output=True, text=True
    ).stdout.splitlines()
    data_changed = any(f.startswith("data/") for f in changed)
    print(f"{ACT} {n} new commit(s) on GitHub" + (" (including data updates)" if data_changed else " (code only)"))
    return ["git pull"] + (INGEST_COMMANDS if data_changed else ["# then restart the server"])


def check_sources_vs_repo() -> list[str]:
    print("\n[3] Live sources vs repo data")
    actions = []

    # cards: compare the card-id set from the API against the local file
    # (id-level check: catches new sets/printings; a pure text edit to an
    # existing card is only caught by running the fetch script)
    try:
        raw = json.loads(get("https://api.dotgg.gg/cgfw/getcards?game=riftbound&mode=indexed", 60))
        id_col = raw["names"].index("id")
        remote_ids = {r[id_col] for r in raw["data"]}
        local = json.loads(Path("data/riftbound_cards.json").read_text(encoding="utf-8"))
        local_ids = {c["id"] for c in local}
        if remote_ids != local_ids:
            print(f"{ACT} card database changed upstream "
                  f"({len(remote_ids - local_ids)} new / {len(local_ids - remote_ids)} removed)")
            actions.append("python scripts/fetch_riftbound_cards.py")
        else:
            print(f"{OK} cards: same {len(remote_ids)} card ids upstream (id-level check)")
    except Exception as e:
        print(f"{WARN} cards check failed: {e}")

    # faq: page list from the sitemap vs pages we have
    try:
        xml = get("https://www.riftboundfaq.com/sitemap.xml")
        remote_pages = set(re.findall(r"<loc>([^<]+)</loc>", xml))
        local_pages = {s["url"] for s in json.loads(Path("data/riftbound_faq.json").read_text(encoding="utf-8"))}
        if remote_pages - local_pages:
            print(f"{ACT} FAQ has {len(remote_pages - local_pages)} new page(s)")
            actions.append("python scripts/fetch_riftbound_faq.py")
        elif local_pages - remote_pages:
            print(f"{ACT} FAQ removed {len(local_pages - remote_pages)} page(s)")
            actions.append("python scripts/fetch_riftbound_faq.py")
        else:
            print(f"{OK} FAQ: same {len(remote_pages)} pages (page-level check)")
    except Exception as e:
        print(f"{WARN} FAQ check failed: {e}")

    # rules: hub link set vs urls in local data (PDF urls are content-hashed)
    try:
        html = get("https://playriftbound.com/en-us/rules-hub/")
        remote_urls = set(re.findall(r'https://[^"\\\s]+\.pdf', html))
        remote_urls |= {
            u for u in re.findall(r'https://riftbound\.leagueoflegends\.com/en-us/news/[^"\\\s]+', html)
            if re.search(r"errata|patch-notes", u)
        }
        remote_urls = {u.rstrip("/") for u in remote_urls}
        local_urls = {s["url"].rstrip("/") for s in json.loads(Path("data/riftbound_rules.json").read_text(encoding="utf-8"))}
        if remote_urls != local_urls:
            new = len(remote_urls - local_urls)
            gone = len(local_urls - remote_urls)
            print(f"{ACT} rules hub changed ({new} new / {gone} removed document link(s))")
            actions.append("python scripts/fetch_riftbound_rules.py")
        else:
            print(f"{OK} rules hub: all {len(remote_urls)} document links unchanged")
    except Exception as e:
        print(f"{WARN} rules check failed: {e}")

    if actions:
        actions.extend(INGEST_COMMANDS)
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/riftbound_kb.db")
    parser.add_argument("--offline", action="store_true", help="Skip the live-source checks.")
    args = parser.parse_args()

    actions = []
    actions += check_repo_vs_remote()
    actions += check_kb_vs_data(args.db)
    if args.offline:
        print("\n[3] Live sources — skipped (--offline)")
    else:
        actions += check_sources_vs_repo()

    actions = list(dict.fromkeys(actions))
    print()
    if not actions:
        print("Everything is up to date.")
        return 0
    print("To update, run:")
    for cmd in actions:
        print(f"  {cmd}")
    print("  # then restart the server")
    return 1


if __name__ == "__main__":
    sys.exit(main())
