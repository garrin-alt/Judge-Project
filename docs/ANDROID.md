# Running ragkb on Android (fully offline)

ragkb runs natively on Android inside [Termux](https://termux.dev) — the
server runs on the phone and you use the web UI in your phone's browser
at `http://localhost:8000`. After the one-time setup below, everything
works with airplane mode on.

The phone profile uses the llama.cpp embeddings backend
(`RAGKB_EMBED_BACKEND=llama`) instead of the desktop default (fastembed/
onnxruntime, which has no Android build). Same embedding model, different
runtime — retrieval quality is identical.

**Space needed:** ~2 GB with answer generation, ~600 MB retrieval-only.
**Recommended:** retrieval-only. It's instant, battery-friendly, and for
rules questions the verbatim rule text with section numbers is more
trustworthy than a small model's paraphrase.

## 1. Install Termux

Install **Termux from F-Droid** (https://f-droid.org/packages/com.termux/).
Do not use the Play Store version — it's abandoned and broken.

## 2. Install packages

In Termux:

```bash
pkg update && pkg upgrade
pkg install python python-numpy git cmake clang make binutils rust
```

(`python-numpy` comes prebuilt from the Termux repo; installing numpy via
pip would try to compile it from source.)

## 3. Get the code and install

```bash
git clone -b claude/rag-knowledge-base-wqbbqv https://github.com/garrin-alt/Judge-Project.git
cd Judge-Project
pip install -e ".[local]"
```

`llama-cpp-python` and `pydantic-core` compile from source here (C++ and
Rust respectively) — expect 15–30 minutes on a phone. This is the only
slow step and only happens once.

Then make the llama embeddings backend the default for this device:

```bash
echo 'export RAGKB_EMBED_BACKEND=llama' >> ~/.bashrc
export RAGKB_EMBED_BACKEND=llama
```

## 4. Build the knowledge base (needs network, once)

```bash
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db
```

The first command downloads the embedding model (~35 MB) automatically.
The whole build takes a few minutes on a modern phone. The card/FAQ/rules
data ships in the repo, so no scraping happens on the phone.

Optional — offline answer generation (adds ~1 GB and slower answers;
skippable if you use "Matches only" mode):

```bash
ragkb download-model
```

## 5. Run it

```bash
termux-wake-lock          # stop Android from killing the server
ragkb --db data/riftbound_kb.db serve
```

Open **http://localhost:8000** in Chrome/Firefox on the phone. Add a home
screen shortcut for one-tap access. From now on this works fully offline.

To stop: Ctrl+C in Termux, then `termux-wake-unlock`.

### Optional: start on boot

Install the **Termux:Boot** add-on (also F-Droid), then:

```bash
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/ragkb.sh <<'EOF'
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
export RAGKB_EMBED_BACKEND=llama
cd ~/Judge-Project
.venv/bin/ragkb --db data/riftbound_kb.db serve 2>> ~/ragkb.log || \
  ragkb --db data/riftbound_kb.db serve 2>> ~/ragkb.log
EOF
chmod +x ~/.termux/boot/ragkb.sh
```

## Updating the knowledge base

When new sets or rules updates drop, on Wi-Fi:

```bash
cd ~/Judge-Project
git pull
pip install -e ".[scrape]"           # once, for the fetch scripts
python scripts/fetch_riftbound_cards.py
python scripts/fetch_riftbound_faq.py
python scripts/fetch_riftbound_rules.py
python scripts/ingest_cards_to_kb.py
python scripts/ingest_faq_to_kb.py
python scripts/ingest_rules_to_kb.py
python scripts/build_glossary.py --db data/riftbound_kb.db
```

## Troubleshooting

- **`pip install fastembed` fails**: expected on Android — don't install
  it. The llama backend replaces it; make sure `RAGKB_EMBED_BACKEND=llama`
  is exported.
- **Server dies when the screen locks**: run `termux-wake-lock`, and
  exempt Termux from battery optimization (Android Settings → Apps →
  Termux → Battery → Unrestricted).
- **Mixed/empty results after switching backends**: embeddings in the DB
  must match the query backend. Delete `data/riftbound_kb.db` and rebuild
  (step 4) with `RAGKB_EMBED_BACKEND=llama` set.
- **Out of memory during generation**: use retrieval-only mode ("Matches
  only" in the UI), or try a smaller GGUF via `RAGKB_LOCAL_MODEL_URL`.
