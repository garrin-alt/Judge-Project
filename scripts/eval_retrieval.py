#!/usr/bin/env python3
"""Measure retrieval quality against tests/eval_questions.json.

Runs each question through the real query pipeline and reports recall@k
and MRR (mean reciprocal rank), overall and per question kind. Use it to
decide whether a retrieval change actually helped: record the numbers
before and after.

    python scripts/eval_retrieval.py                    # summary
    python scripts/eval_retrieval.py --misses           # list failures
    python scripts/eval_retrieval.py --save before.json # snapshot
    python scripts/eval_retrieval.py --compare before.json

A case passes at rank i if any of its `expect` substrings appears in the
i-th retrieved result (case-insensitive) — in the title by default, or in
the body text when the case sets `"in": "text"` (use that when the signal
lives in the content, e.g. a card's colour line, since rule titles carry
section breadcrumbs rather than rule names).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ragkb.query import run_query
from ragkb.store import KnowledgeStore

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "tests" / "eval_questions.json"


def hit_rank(expect: list[str], haystacks: list[str]) -> int | None:
    """1-based rank of the first result matching any expected substring."""
    wanted = [e.lower() for e in expect]
    for i, hay in enumerate(haystacks, start=1):
        low = hay.lower()
        if any(w in low for w in wanted):
            return i
    return None


def evaluate(db: str, cases: list[dict], top_k: int) -> dict:
    results = []
    with KnowledgeStore(db) as store:
        for case in cases:
            resp = run_query(store, case["q"], top_k=top_k)
            titles = [r.title for r in resp.results]
            haystacks = ([r.text for r in resp.results]
                         if case.get("in") == "text" else titles)
            rank = hit_rank(case["expect"], haystacks)
            results.append({
                "q": case["q"],
                "kind": case.get("kind", "other"),
                "expect": case["expect"],
                "rank": rank,
                "titles": titles,
            })
    return {"top_k": top_k, "results": results}


def summarize(report: dict) -> dict:
    results = report["results"]
    k = report["top_k"]

    def stats(rows):
        n = len(rows)
        if not n:
            return {}
        at1 = sum(1 for r in rows if r["rank"] == 1) / n
        at3 = sum(1 for r in rows if r["rank"] and r["rank"] <= 3) / n
        atk = sum(1 for r in rows if r["rank"]) / n
        mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
        return {"n": n, "recall@1": at1, "recall@3": at3, f"recall@{k}": atk, "mrr": mrr}

    by_kind = defaultdict(list)
    for r in results:
        by_kind[r["kind"]].append(r)
    return {"overall": stats(results), "by_kind": {kind: stats(rows) for kind, rows in sorted(by_kind.items())}}


def print_summary(summary: dict, k: int):
    o = summary["overall"]
    print(f"\n{'':14} {'n':>4} {'R@1':>7} {'R@3':>7} {f'R@{k}':>7} {'MRR':>7}")
    print("-" * 50)
    for kind, s in summary["by_kind"].items():
        print(f"{kind:14} {s['n']:>4} {s['recall@1']:>7.1%} {s['recall@3']:>7.1%} {s[f'recall@{k}']:>7.1%} {s['mrr']:>7.3f}")
    print("-" * 50)
    print(f"{'OVERALL':14} {o['n']:>4} {o['recall@1']:>7.1%} {o['recall@3']:>7.1%} {o[f'recall@{k}']:>7.1%} {o['mrr']:>7.3f}")


def print_compare(old: dict, new: dict, k: int):
    a, b = summarize(old)["overall"], summarize(new)["overall"]
    print(f"\n{'metric':14} {'before':>9} {'after':>9} {'delta':>9}")
    print("-" * 45)
    for key in ["recall@1", "recall@3", f"recall@{k}", "mrr"]:
        if key in a and key in b:
            fmt = (lambda v: f"{v:.1%}") if key.startswith("recall") else (lambda v: f"{v:.3f}")
            delta = b[key] - a[key]
            print(f"{key:14} {fmt(a[key]):>9} {fmt(b[key]):>9} {delta:>+9.3f}")

    old_rank = {r["q"]: r["rank"] for r in old["results"]}
    changed = [(r["q"], old_rank.get(r["q"]), r["rank"]) for r in new["results"]
               if r["q"] in old_rank and old_rank[r["q"]] != r["rank"]]
    if changed:
        print("\nper-question changes (rank; None = not found):")
        for q, before, after in changed:
            better = (after or 99) < (before or 99)
            print(f"  {'+' if better else '-'} {before} -> {after}  {q}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/riftbound_kb.db")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--misses", action="store_true", help="Show questions with no hit in top-k.")
    parser.add_argument("--save", help="Write the full report to this path.")
    parser.add_argument("--compare", help="Compare against a previously saved report.")
    args = parser.parse_args()

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))["cases"]
    print(f"Evaluating {len(cases)} questions against {args.db} (top_k={args.top_k})...")
    report = evaluate(args.db, cases, args.top_k)

    print_summary(summarize(report), args.top_k)

    if args.misses:
        misses = [r for r in report["results"] if r["rank"] is None]
        print(f"\n{len(misses)} miss(es):")
        for r in misses:
            print(f"\n  Q: {r['q']}")
            print(f"     expected any of: {r['expect']}")
            for t in r["titles"]:
                print(f"     got: {t}")

    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report to {args.save}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        print_compare(old, report, args.top_k)


if __name__ == "__main__":
    main()
