# ragkb

A small, local RAG (retrieval-augmented generation) knowledge base with a
CLI and a web UI. Store notes, documents, or any text as a knowledge base,
then retrieve relevant pieces with a short prompt — optionally getting a
synthesized answer grounded in what you stored, generated fully offline by
a local model (or by Claude via the Anthropic API).

- **Storage**: local SQLite database (default `~/.ragkb/kb.db`)
- **Embeddings**: local, offline model — no per-call API key or cost. Two
  backends: [fastembed](https://github.com/qdrant/fastembed) (desktop
  default, install with the `[onnx]` extra) or the same model as GGUF via
  llama.cpp (`RAGKB_EMBED_BACKEND=llama`, works on Android/Termux — see
  [docs/ANDROID.md](docs/ANDROID.md)). The model downloads once on first
  use and is cached locally.
- **Generation**: two interchangeable backends —
  - `local`: fully offline via [llama.cpp](https://github.com/ggml-org/llama.cpp) (default model: Qwen2.5-1.5B-Instruct, ~1 GB one-time download)
  - `claude`: Anthropic API (requires `ANTHROPIC_API_KEY`)
- **Interfaces**: CLI, REST API, and a single-page web UI (`ragkb serve`)

## Install

Sharing with friends? Send them **[docs/INSTALL.md](docs/INSTALL.md)** — one page covering computer, Android, and no-install options.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[onnx]"
```

## Usage

### Add content to the knowledge base

```bash
ragkb add --file notes.txt --title "Project notes"
ragkb add --text "The standup is at 9:30am PT every weekday." --title "Standup time"
```

Text is split into overlapping chunks, embedded locally, and stored in SQLite.

### Query it

```bash
ragkb query "when is the standup?"
```

This retrieves the most relevant stored chunks and, if `ANTHROPIC_API_KEY`
is set, asks Claude to answer the question using only that retrieved
context. Without an API key (or with `--no-generate`), it just prints the
raw matching chunks:

```bash
ragkb query "when is the standup?" --no-generate
```

### Web UI

```bash
pip install -e ".[local]"   # adds llama-cpp-python for offline answers
ragkb download-model         # one-time ~1 GB model download
ragkb serve                  # open http://127.0.0.1:8000/
```

The UI has a search tab (retrieval with optional generated answers — pick
"local model" for fully offline operation) and a documents tab for adding
and removing knowledge-base entries. The same server exposes a REST API:
`GET /api/status`, `GET|POST /api/documents`, `DELETE /api/documents/{id}`,
`POST /api/query`.

### Manage the knowledge base

```bash
ragkb list             # list stored documents
ragkb remove <id>       # remove a document by id
ragkb reset             # clear everything (asks for confirmation)
```

## Configuration

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `RAGKB_DB_PATH` | `~/.ragkb/kb.db` | Where the SQLite database lives |
| `RAGKB_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model name |
| `RAGKB_EMBED_BACKEND` | `fastembed` | `fastembed` (onnx) or `llama` (GGUF via llama.cpp) |
| `RAGKB_EMBED_MODEL_URL` | bge-small-en-v1.5 Q8_0 GGUF | Embedding model for the `llama` backend |
| `RAGKB_ANTHROPIC_MODEL` | `claude-sonnet-5` | Model used by the `claude` backend |
| `RAGKB_LOCAL_MODEL_URL` | Qwen2.5-1.5B-Instruct Q4_K_M | GGUF download URL for the `local` backend |
| `RAGKB_MODEL_DIR` | `~/.ragkb/models` | Where local model files are cached |
| `RAGKB_CHUNK_SIZE` | `800` | Max characters per chunk |
| `RAGKB_CHUNK_OVERLAP` | `100` | Character overlap between chunks |
| `ANTHROPIC_API_KEY` | — | Required only for `query`'s answer-generation step |

You can also point at a different database per-command with `--db path/to.db`.

## How it works

1. `add` splits input text into overlapping chunks (`ragkb/chunking.py`),
   embeds each chunk locally with fastembed (`ragkb/embeddings.py`), and
   stores the text + embedding vector in SQLite (`ragkb/store.py`).
2. `query` embeds your prompt the same way, computes cosine similarity
   against every stored chunk, and returns the top matches.
3. If an Anthropic API key is available, the top matches are passed to
   Claude as context so it can produce a direct, grounded answer
   (`ragkb/generate.py`).

## Example dataset: Riftbound TCG cards

The repo includes a worked example — the full Riftbound card database
(card text and tags, as browsable on riftbound.gg):

- `data/riftbound_cards.json` / `.csv` — 1,147 collated cards (name, rules
  text, tags, cost, colors, set, rarity; no imagery or prices)
- `data/riftbound_faq.json` — 92 judge-ruling Q&A sections collated from
  [riftboundfaq.com](https://riftboundfaq.com) (rules interactions, chain
  and priority, mechanics, per-card rulings)
- `data/riftbound_rules.json` — 475 sections of official rules from the
  [playriftbound.com rules hub](https://playriftbound.com/en-us/rules-hub/):
  the Core Rules (354 numbered rules) and Tournament Rules (66 rules)
  PDFs split per rule, plus errata and patch-notes articles
- `scripts/fetch_riftbound_cards.py` — re-fetches the card dataset from
  the card API that riftbound.gg's browser uses
- `scripts/fetch_riftbound_faq.py` — re-scrapes the FAQ site (discovers
  pages via its sitemap, splits articles into Q&A sections)
- `scripts/ingest_cards_to_kb.py` — loads the cards into a ragkb knowledge
  base (one document per card, reprints collapsed)
- `scripts/fetch_riftbound_rules.py` — re-fetches the official documents
  (discovers the current PDF/article links from the rules hub page)
- `scripts/ingest_faq_to_kb.py` / `scripts/ingest_rules_to_kb.py` — add
  those datasets to the same KB (re-runnable; each replaces its own
  previously ingested entries)

```bash
python scripts/ingest_cards_to_kb.py            # builds data/riftbound_kb.db
python scripts/ingest_faq_to_kb.py              # adds the community rules FAQ
python scripts/ingest_rules_to_kb.py            # adds official rules/errata/patch notes
python scripts/build_glossary.py --db data/riftbound_kb.db   # game-term glossary
ragkb --db data/riftbound_kb.db query "which cards counter spells?"
ragkb --db data/riftbound_kb.db query "when do triggered abilities trigger?"
ragkb --db data/riftbound_kb.db query "what happens in rule 354 when playing a card?"
```

The rules are stored hierarchically: section headings become breadcrumbs on
each rule ("Riftbound Core Rules §344 — Showdowns"), and short sections
(e.g. step-by-step procedures) are kept together as one document.

`build_glossary.py` extracts game vocabulary from the Core Rules — the six
Domains with their colors (rule 134), keyword definitions (Assault,
Ambush, ...), and definition-style rules — and installs it into the KB.
The query pipeline (`ragkb/query.py`) uses it to annotate prompts
("purple champion" → "purple (Chaos) champion") before embedding and to
hand matched definitions to the answer model.

The pipeline also does **hybrid card search**: prompts with hard
constraints (domains/colors, cost/might comparisons, card types,
keywords, tags — e.g. "fury units with Assault costing 2 or less") are
parsed into SQL filters over a structured cards table and only then
ranked semantically. Impossible combinations report "no cards match"
with the parsed filters instead of returning look-alikes. Retrieved
rules and FAQ answers additionally pull in the rules they cite
("See rule 307.", "[354.2]") as labeled referenced-rule context.

Queries now blend both sources — asking about a card like Arcane Shift
returns its card text alongside the judge rulings about how it resolves.

## Development

```bash
pip install -e ".[dev]"
pytest
```

### Measuring retrieval quality

`tests/eval_questions.json` holds questions with the documents they should
retrieve; `scripts/eval_retrieval.py` reports recall@k and MRR overall and
per question kind. Record a baseline before a retrieval change and compare
after — several plausible-sounding ideas measurably hurt this corpus:

```bash
python scripts/eval_retrieval.py --save before.json
# ...make a change...
python scripts/eval_retrieval.py --compare before.json
```

Note: the first `add`/`query` call downloads the embedding model from
Hugging Face and caches it locally — it needs network access once, after
which everything runs fully offline.
