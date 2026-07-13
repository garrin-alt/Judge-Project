# Installing the Riftbound Judge Knowledge Base

An offline, searchable knowledge base for Riftbound: every card, the
community judge FAQ, and the official Core Rules, Tournament Rules,
errata, and patch notes — cross-referenced and queryable in plain
English from a web UI. After setup it works with no internet at all.

Three ways to use it, easiest first.

---

## Option 1 — No install: join someone who runs it

If a friend runs the server at an event, you need nothing but a browser:

1. Join their Wi-Fi network or phone hotspot.
2. Open the address they give you (e.g. `http://192.168.43.1:8000`).

That's it. Bookmark it / "Add to Home screen" for one-tap access.

---

## Option 2 — Computer (Mac, Linux, Windows)

Needs Python 3.10+ and ~2 GB disk (600 MB without answer generation).

```bash
git clone -b claude/rag-knowledge-base-wqbbqv https://github.com/garrin-alt/Judge-Project.git
cd Judge-Project
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[onnx,local]"

# build the knowledge base (data ships in the repo; embeds locally, a few minutes)
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db

# optional: offline AI answers (~1 GB one-time download; skippable)
ragkb download-model

# run it
ragkb --db data/riftbound_kb.db serve
```

Open **http://127.0.0.1:8000**. To share it with other devices on your
network, run `ragkb --db data/riftbound_kb.db serve --host 0.0.0.0` and
give people `http://<your-ip>:8000`.

Windows note: use PowerShell; if `python3` isn't found, use `python`.

---

## Option 3 — Android phone (fully offline, in your pocket)

Runs natively inside Termux. Full walkthrough with troubleshooting:
**[docs/ANDROID.md](ANDROID.md)**. Condensed version:

1. Install **Termux from F-Droid** (not the Play Store).
2. In Termux:

```bash
pkg update && pkg upgrade
pkg install python python-numpy git cmake clang make binutils rust ninja libcurl
git clone -b claude/rag-knowledge-base-wqbbqv https://github.com/garrin-alt/Judge-Project.git
cd Judge-Project
pip install setuptools wheel scikit-build-core
CMAKE_ARGS="-DLLAMA_CURL=OFF -DGGML_OPENMP=OFF" pip install llama-cpp-python --no-build-isolation
pip install -e ".[local]"
echo 'export RAGKB_EMBED_BACKEND=llama' >> ~/.bashrc
export RAGKB_EMBED_BACKEND=llama
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db
termux-wake-lock
ragkb --db data/riftbound_kb.db serve
```

3. Open **http://localhost:8000** in your phone browser.

The `llama-cpp-python` compile takes 10–20 minutes on a phone — one
time. If anything errors, the troubleshooting section of
[ANDROID.md](ANDROID.md) covers every failure we've seen (broken
Termux packages, the "Unsupported platform" import error, CMake issues).

---

## Using it

- Type a question the way you'd ask a judge: *"remedy for drawing an
  extra card"*, *"can I react to triggered abilities?"*, *"fury units
  with Assault costing 2 or less"*.
- **Matches only** (default) returns the exact card text / rule text
  with section numbers — instant and authoritative. The colored badge
  on each result tells you the source (Card, FAQ, Core rules,
  Tournament, Errata, Patch notes).
- **Filter chips** under the search box restrict to specific sources.
- **Answer modes** generate a plain-English answer from the retrieved
  text (needs the optional model download). Treat it as a convenience —
  for actual rulings, read the retrieved rule text.
- Penalties in tournament text are highlighted: Warning / Game Loss /
  Disqualification / No Penalty.
- **Documents tab**: add your own notes (store policies, event rules) —
  they become searchable alongside everything else.

## Updating when new sets or rules drop

```bash
cd Judge-Project
git pull
pip install -e ".[scrape]"                     # once
python scripts/fetch_riftbound_cards.py
python scripts/fetch_riftbound_faq.py
python scripts/fetch_riftbound_rules.py
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db
```

Data sources: [riftbound.gg](https://riftbound.gg) (cards),
[riftboundfaq.com](https://riftboundfaq.com) (community judge FAQ),
[playriftbound.com rules hub](https://playriftbound.com/en-us/rules-hub/)
(official rules, errata, patch notes). All content belongs to its
respective owners; this tool just makes it searchable for personal use.
