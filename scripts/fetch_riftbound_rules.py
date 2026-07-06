#!/usr/bin/env python3
"""Fetch and collate official Riftbound rules documents from playriftbound.com.

Discovers the current documents from the rules hub page:
- Core Rules PDF and Tournament Rules PDF (split into numbered rule sections,
  so FAQ citations like [354.2] can be looked up directly)
- Errata and patch-notes articles (split into heading sections)

Usage:
    python scripts/fetch_riftbound_rules.py [--out data/riftbound_rules.json]
"""

import argparse
import io
import json
import re
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

HUB_URL = "https://playriftbound.com/en-us/rules-hub/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# Top-level rule boundary: "354. Text" (sub-rules like "354.2.a." have no space
# after the first number, so they stay attached). Inline citations ("see rule
# 307.", "in section 700.") are excluded by the lookbehinds; any that slip
# through are dropped by the increasing-subsequence filter in parse_pdf.
_CITE_WORDS = ["rule", "rules", "section", "sections", "step", "steps", "see", "in"]
RULE_RE = re.compile(
    r"(?:(?<=\s)|^)"
    + "".join(f"(?<!{w} )(?<!{w.capitalize()} )" for w in _CITE_WORDS)
    + r"(\d{3})\.\s+"
)


def _longest_increasing(matches: list) -> list:
    """Longest non-decreasing subsequence of rule-number matches (O(n^2))."""
    if not matches:
        return []
    nums = [int(m.group(1)) for m in matches]
    best_len = [1] * len(nums)
    prev = [-1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] <= nums[i] and best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j
    i = max(range(len(nums)), key=lambda k: best_len[k])
    chain = []
    while i != -1:
        chain.append(matches[i])
        i = prev[i]
    return chain[::-1]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def discover_links() -> tuple[list[str], list[str]]:
    html = get(HUB_URL).decode("utf-8")
    pdfs = sorted(set(re.findall(r'https://[^"\\\s]+\.pdf', html)))
    articles = sorted(set(
        u for u in re.findall(r'https://riftbound\.leagueoflegends\.com/en-us/news/[^"\\\s]+', html)
        if re.search(r"errata|patch-notes", u)
    ))
    return pdfs, articles


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(url: str) -> list[dict]:
    reader = PdfReader(io.BytesIO(get(url)))
    full = normalize(" ".join(page.extract_text() or "" for page in reader.pages))

    title_match = re.match(r"(Riftbound\s+\w+\s+Rules)", full)
    doc_name = title_match.group(1) if title_match else url.rsplit("/", 1)[-1]

    # Rule numbers ascend through the document; keep the longest consistent
    # chain so stray citation numbers inside rule text are dropped.
    matches = _longest_increasing(list(RULE_RE.finditer(full)))

    sections = []
    for i, m in enumerate(matches):
        number = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        body = full[m.end():end].strip()
        if not body:
            continue
        sections.append({
            "doc": doc_name,
            "section": number,
            "title": f"{doc_name} §{number}",
            "text": f"{number}. {body}",
            "url": url,
        })
    return sections


def parse_article(url: str) -> list[dict]:
    soup = BeautifulSoup(get(url).decode("utf-8"), "html.parser")
    main = soup.find("article") or soup.find("main")
    if main is None:
        return []
    h1 = main.find("h1")
    doc_name = normalize(h1.get_text(" ", strip=True)) if h1 else url

    sections = []
    heading = "Overview"
    parts = []

    def flush():
        text = normalize(" ".join(parts))
        if text:
            sections.append({
                "doc": doc_name,
                "section": heading,
                "title": f"{doc_name} — {heading}",
                "text": text,
                "url": url,
            })

    for el in main.find_all(["h2", "h3", "p", "li", "td", "th"]):
        text = normalize(el.get_text(" ", strip=True))
        if not text:
            continue
        if el.name in ("h2", "h3"):
            flush()
            heading = text
            parts = []
        else:
            parts.append(text)
    flush()
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/riftbound_rules.json")
    args = parser.parse_args()

    pdfs, articles = discover_links()
    print(f"Discovered {len(pdfs)} PDF(s) and {len(articles)} article(s) on the rules hub")

    all_sections = []
    for url in pdfs:
        sections = parse_pdf(url)
        print(f"  {sections[0]['doc'] if sections else url}: {len(sections)} rule section(s)")
        all_sections.extend(sections)
    for url in articles:
        sections = parse_article(url)
        print(f"  {url.rstrip('/').rsplit('/', 1)[-1]}: {len(sections)} section(s)")
        all_sections.extend(sections)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_sections, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_sections)} sections to {out}")


if __name__ == "__main__":
    main()
