#!/usr/bin/env python3
"""Fetch and collate riftboundfaq.com (community Riftbound rules FAQ).

Discovers all pages from the sitemap, extracts each page's article content,
and splits it into Q&A sections (one per question heading). Output is a
JSON list of sections ready for knowledge-base ingestion.

Usage:
    python scripts/fetch_riftbound_faq.py [--out data/riftbound_faq.json]
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

SITEMAP_URL = "https://www.riftboundfaq.com/sitemap.xml"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

# UI chrome that appears inside the article but isn't content.
CHROME_TEXTS = {"Edit this page", "Feedback", "On this page"}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def page_urls() -> list[str]:
    xml = get(SITEMAP_URL)
    return re.findall(r"<loc>([^<]+)</loc>", xml)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_sections(url: str, html: str) -> list[dict]:
    """Split an article into (page title, question, answer) sections."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article is None:
        return []

    h1 = article.find("h1")
    page_title = clean(h1.get_text(" ", strip=True)) if h1 else url

    sections = []
    current_heading = None
    current_parts = []

    def flush():
        text = clean(" ".join(current_parts))
        if current_heading and text:
            sections.append({
                "page": page_title,
                "question": current_heading,
                "answer": text,
                "url": url,
            })

    for el in article.find_all(["h2", "h3", "p", "li", "blockquote", "td", "th"]):
        text = clean(el.get_text(" ", strip=True))
        if not text or text in CHROME_TEXTS:
            continue
        if el.name in ("h2", "h3"):
            flush()
            current_heading = text
            current_parts = []
        elif current_heading:
            current_parts.append(text)
    flush()

    # Pages without question headings (e.g. the About page) become one section.
    if not sections:
        body = clean(" ".join(
            clean(el.get_text(" ", strip=True))
            for el in article.find_all(["p", "li"])
            if clean(el.get_text(" ", strip=True)) not in CHROME_TEXTS
        ))
        if body:
            sections.append({"page": page_title, "question": page_title, "answer": body, "url": url})
    return sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/riftbound_faq.json")
    args = parser.parse_args()

    urls = page_urls()
    print(f"{len(urls)} pages in sitemap")

    all_sections = []
    for url in urls:
        html = get(url)
        sections = extract_sections(url, html)
        print(f"  {url} -> {len(sections)} section(s)")
        all_sections.extend(sections)
        time.sleep(0.3)  # be polite

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_sections, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_sections)} Q&A sections to {out}")


if __name__ == "__main__":
    main()
