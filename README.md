# ragkb

A small, local RAG (retrieval-augmented generation) knowledge base. Store
notes, documents, or any text as a knowledge base, then retrieve relevant
pieces with a short prompt — optionally getting a synthesized answer back
from Claude, grounded in what you stored.

- **Storage**: local SQLite database (default `~/.ragkb/kb.db`)
- **Embeddings**: local, offline model via [fastembed](https://github.com/qdrant/fastembed) — no per-call API key or cost. (The model itself is downloaded from Hugging Face once on first use and then cached locally.)
- **Generation**: optional, uses the Anthropic API (Claude) to answer questions grounded in retrieved chunks

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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
| `RAGKB_ANTHROPIC_MODEL` | `claude-sonnet-5` | Model used for answer generation |
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
- `scripts/fetch_riftbound_cards.py` — re-fetches the dataset from the
  card API that riftbound.gg's browser uses
- `scripts/ingest_cards_to_kb.py` — loads the cards into a ragkb knowledge
  base (one document per card, reprints collapsed)

```bash
python scripts/ingest_cards_to_kb.py            # builds data/riftbound_kb.db
ragkb --db data/riftbound_kb.db query "which cards counter spells?"
ragkb --db data/riftbound_kb.db query "yordle units that draw cards"
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Note: the first `add`/`query` call downloads the embedding model from
Hugging Face and caches it locally — it needs network access once, after
which everything runs fully offline.
