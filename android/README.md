# Riftbound Judge (Android)

Offline, on-device rules lookup for Riftbound judges. No network calls, no API
keys — judges need rulings mid-match on venue wifi that can't be relied on.

Forked from [PocketSage](https://github.com/umarpazir11/pocketsage) (MIT); see
[LICENSE](LICENSE). The RAG pipeline is PocketSage's: import a document, chunk and
embed it locally, top-K cosine retrieval from a Room store, stream answers from a
Gemma `.litertlm` model via LiteRT-LM.

## Status

Early. Step 1 of 6 — the fork is renamed, ABI-restricted and building clean.
Rules-document ingest and citation precision are not built yet.

## Build

```bash
export ANDROID_HOME=/path/to/android-sdk
./gradlew testDebugUnitTest lintDebug assembleDebug
```

Requires JDK 21 and Android SDK 35. `minSdk` is 26, `targetSdk` 35.

The APK is restricted to **arm64-v8a** — LiteRT-LM ships native libraries for
arm64-v8a and x86_64 only, so a 32-bit build would install and then fail to link.
A physical arm64 device is required: emulators lack the acceleration needed for
on-device LLM inference, and neither the embedder nor LiteRT-LM can be verified
on one.

## Models

Two models, provisioned differently.

**Embedding** — `all-MiniLM-L6-v2.tflite` (22 MB) ships in `app/src/main/assets/`.
Note its sequence length is capped at 128 tokens, so a chunk longer than roughly
400–500 characters is stored but partly invisible to search.

**Generation** — a Gemma `.litertlm` (~2.6 GB) is **never bundled in the APK**. It
is imported at runtime through the Storage Access Framework into app-private
storage, copied via a `.tmp` file and renamed only once complete.

## Known issues inherited from upstream

- `TfLiteEmbeddingService` uses a whole-word tokenizer against a WordPiece vocab,
  so out-of-vocab terms become `[UNK]` and `702.12.b.2` shreds into four tokens.
  The correct `BertTokenizer` exists and is tested but has no callers. Fixing this
  is step 2.
- `Chunker` is a fixed 800-character window with no structural awareness.
- `RetrievedChunk` carries no title, page or section number, so answers cannot
  cite a rule.
- `RagDatabase` is version 1 with `exportSchema = false` and no migration path.
