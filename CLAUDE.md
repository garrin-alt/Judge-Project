# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# setup (desktop)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[onnx,local,dev,scrape]"

pytest                                   # all tests
pytest tests/test_cardsearch.py -q       # one file
pytest tests/test_store.py::test_search_returns_one_result_per_document   # one test

# build the knowledge base from the committed data/*.json (a few minutes)
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db

# run
ragkb --db data/riftbound_kb.db query "what is the penalty for slow play?" --no-generate
ragkb --db data/riftbound_kb.db serve          # http://127.0.0.1:8000
ragkb download-model                            # ~1GB GGUF, only for generated answers

# freshness: KB vs data files, repo vs GitHub, live sources vs repo
python scripts/check_updates.py [--offline]
```

## Measure retrieval changes — always

`scripts/eval_retrieval.py` scores `tests/eval_questions.json` (recall@1/3/k, MRR, by
question kind). **Record a baseline before touching retrieval and compare after:**

```bash
python scripts/eval_retrieval.py --save before.json
python scripts/eval_retrieval.py --compare before.json --misses
```

This is not ceremony. Three techniques that standard RAG guidance recommends were
measured on this corpus and **made it worse**, and each is now off by default with its
numbers recorded in `ragkb/config.py`:

- bge query-instruction prefix (`QUERY_PREFIX`): recall@1 93.8% → 85.4%
- broad BM25 hybrid fusion (`LEXICAL_WEIGHT` over ordinary words): monotonically worse at every weight
- cross-encoder reranking (`RERANK_BACKEND`): 94.2% → 91.4% (ms-marco), 84.5% (bge-reranker)

The corpus is short structured records (card entries, numbered rule fragments), not prose
passages; the first stage already encodes the signal structurally. Generic advice tuned for
prose tends to lose here. Assume nothing works until the harness says so.

## Architecture

Two layers that must stay separate:

- **`ragkb/`** — a general-purpose local RAG library. Nothing Riftbound-specific belongs here.
- **`scripts/` + `data/`** — the Riftbound application: scrapers, ingest, glossary build.

### The query pipeline is the single choke point

`ragkb/query.py::run_query` is what CLI, REST API and web UI all call. Anything added there
lands in all three interfaces at once. Stages, in order:

1. **Glossary expansion** (`glossary.py`) — rewrites user vocabulary into knowledge-base
   vocabulary ("purple champion" → "purple (Chaos) champion") and collects definitions for
   the generator. The glossary is *derived from the rules themselves* by
   `scripts/build_glossary.py` (domains+colours from rule 134, keyword→rule mapping, definition-style rules).
2. **Routing** — a prompt with hard card constraints goes to `cardsearch.py`, which parses
   colours/cost/might/type/keywords/tags into real SQL over the typed `cards` table and only
   then ranks semantically. Impossible combinations report "no cards match" honestly and tell
   the generator to say so, rather than returning look-alikes. Everything else takes the
   semantic path. `triggers_card_search()` is deliberately conservative so rules questions
   that merely mention a keyword don't get hijacked.
3. **Retrieval** (`store.py::search`) — dense cosine over all chunks, one best chunk per
   document, fused via RRF with an FTS5 keyword arm that fires **only on exact identifiers**
   (card codes like `OGN-224`, dotted rule numbers like `702.12.b.2`) — see `store._fts_query`.
4. **Entity anchoring** (`entities.py`) — extracts named cards/rules/keywords and guarantees
   each a slot, so a question naming two cards can't return neither. The best whole-question
   hit still leads: naming a card must not demote the ruling that answers the question.
   Multi-clause questions are also split and retrieved per clause.
5. **Citation expansion** (`citations.py`) — follows "See rule 307." and FAQ `[354.2]`
   references into the cited rule documents, plus card keywords → their defining rules.
6. **Generation** (`generate.py`) — `local` (llama.cpp, in-process) or `claude` (API), with
   `auto` resolution. Retrieval-only is the trustworthy default; the 1.5B local model is
   reliable on lookups and unreliable on multi-step procedures.

### Storage invariants (`store.py`)

One SQLite file holds everything: `documents`, `chunks` (float32 embedding blobs), plus
`cards` (typed columns for filtering), `glossary`, `meta`, and an optional `chunks_fts`
index kept in sync by triggers. Portability is the point — the file can be copied to a phone.

- **Embeddings in a KB must come from the same model+backend that embeds queries.**
  `check_embedding_provenance()` stamps `meta` on first write and warns on mismatch. Switching
  `RAGKB_EMBED_BACKEND` means rebuilding the KB, not just re-running.
- Schema changes need a migration in `_migrate()` (existing phone KBs upgrade in place).
- FTS5 is optional: builds without it degrade to dense-only rather than failing.

### Data flow and regeneration

`fetch_riftbound_*.py` → `data/*.json` (committed) → `ingest_*.py` → `data/riftbound_kb.db`
(gitignored). The JSON is committed so installs need no scraping; the KB is always
rebuildable. Ingest scripts are **idempotent**: each removes its own prior documents (matched
by source URL prefix) before re-inserting, and records a SHA-256 fingerprint of its input in
`meta` so `check_updates.py` can tell whether the KB matches the data.

Chunking is structure-aware, not character-windowed: `ingest_rules_to_kb.py` splits rules at
their own sub-rule numbering (`702.1.`, `702.2.a.`) and never mid-clause, folds section
headings into breadcrumbs on each chunk, and keeps short procedures whole.
`config.MAX_CHUNK_CHARS` exists because text beyond the embedding model's ~512-token window
would be stored but invisible to search.

## Android/Termux constraint

The phone target shapes real design decisions — see `docs/ANDROID.md`. onnxruntime has no
Android build, so `embeddings.py` has two backends producing identical vectors
(`RAGKB_EMBED_BACKEND=fastembed|llama`). Don't add heavy native dependencies, don't break
single-file SQLite portability, and don't delegate query-side text transforms to a
backend-specific helper — both backends must produce the same vector for the same string.

## Conventions

- New dependencies go in an extra in `pyproject.toml`, not core, unless every install needs
  them. Core stays importable on a phone.
- Tests use deterministic fake embeddings (hashlib, not `hash()`, which is seed-randomised).
- When a plausible idea is measured and rejected, keep the mechanism behind a config flag and
  record the numbers in the comment. That comment is the reason nobody re-adds it on principle.
