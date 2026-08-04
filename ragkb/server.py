"""REST API + web UI server for ragkb."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config
from .chunking import chunk_text
from .embeddings import embed_texts
from .generate import (
    SYSTEM_PROMPT,
    GenerationUnavailable,
    build_user_message,
    generate_answer,
    resolve_backend,
)
from .query import run_query
from .store import KnowledgeStore

WEB_DIR = Path(__file__).parent / "web"


class AddDocumentRequest(BaseModel):
    title: str
    text: str
    source: str | None = None


class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5
    # "none" = retrieval only; "auto"/"local"/"claude" = also generate an answer
    generate: str = "none"
    # restrict to document kinds (card/faq/core/tournament/errata/patch/note);
    # empty = all sources
    sources: list[str] = []


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="ragkb", version="0.1.0")
    resolved_db = db_path or str(config.DEFAULT_DB_PATH)

    def open_store() -> KnowledgeStore:
        return KnowledgeStore(resolved_db)

    @app.get("/api/status")
    def status():
        from . import localmodel

        with open_store() as store:
            docs = store.list_documents()
            chunks = store.count_chunks()
        return {
            "db_path": resolved_db,
            "documents": len(docs),
            "chunks": chunks,
            "backends": {
                "local": localmodel.is_available(),
                "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
            },
        }

    @app.get("/api/documents")
    def list_documents():
        with open_store() as store:
            docs = store.list_documents()
        return [
            {"id": d[0], "title": d[1], "source": d[2], "created_at": d[3], "chunks": d[4]}
            for d in docs
        ]

    @app.post("/api/documents")
    def add_document(req: AddDocumentRequest):
        chunks = chunk_text(req.text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
        if not chunks:
            raise HTTPException(status_code=400, detail="Text is empty.")
        vectors = embed_texts(chunks)
        with open_store() as store:
            doc_id = store.add_document(req.title, req.source, chunks, vectors)
        return {"id": doc_id, "chunks": len(chunks)}

    @app.get("/api/documents/{doc_id}/text")
    def document_text(doc_id: int):
        with open_store() as store:
            try:
                title, source, text = store.get_document_text(doc_id, max_chars=None)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"No document #{doc_id}.")
        return {"id": doc_id, "title": title, "source": source, "text": text}

    @app.delete("/api/documents/{doc_id}")
    def remove_document(doc_id: int):
        with open_store() as store:
            removed = store.remove_document(doc_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"No document #{doc_id}.")
        return {"removed": doc_id}

    @app.post("/api/query")
    def query(req: QueryRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is empty.")
        with open_store() as store:
            resp = run_query(
                store, req.prompt, top_k=req.top_k, generate=req.generate,
                sources=req.sources or None,
            )

        with open_store() as store:
            partial = {
                r.doc_id: store.count_doc_chunks(r.doc_id) > 1 for r in resp.results
            }

        return {
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "source": r.source,
                    "text": r.text,
                    "score": round(r.score, 4),
                    "partial": partial.get(r.doc_id, False),
                }
                for r in resp.results
            ],
            "related": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "source": r.source,
                    "text": r.text,
                    "cited_by": r.cited_by,
                }
                for r in resp.related
            ],
            "answer": resp.answer,
            "backend": resp.backend,
            "notice": resp.notice,
            "expanded_prompt": resp.expanded_prompt if resp.expanded_prompt != resp.prompt else None,
            "definitions": resp.definitions,
            "card_filters": resp.card_filters,
        }

    @app.post("/api/query/stream")
    def query_stream(req: QueryRequest):
        """Same as /api/query but server-sent events: retrieval results are
        delivered as soon as they exist, then answer text token by token."""
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is empty.")

        def events():
            import json as _json

            with open_store() as store:
                resp = run_query(store, req.prompt, top_k=req.top_k,
                                 generate="none", sources=req.sources or None)
                partial = {r.doc_id: store.count_doc_chunks(r.doc_id) > 1 for r in resp.results}
                payload = {
                    "results": [
                        {"doc_id": r.doc_id, "title": r.title, "source": r.source,
                         "text": r.text, "score": round(r.score, 4),
                         "partial": partial.get(r.doc_id, False)}
                        for r in resp.results
                    ],
                    "related": [
                        {"doc_id": r.doc_id, "title": r.title, "source": r.source,
                         "text": r.text, "cited_by": r.cited_by}
                        for r in resp.related
                    ],
                    "notice": resp.notice,
                    "expanded_prompt": (resp.expanded_prompt
                                        if resp.expanded_prompt != resp.prompt else None),
                    "definitions": resp.definitions,
                    "card_filters": resp.card_filters,
                }
                yield f"event: results\ndata: {_json.dumps(payload)}\n\n"

                if req.generate == "none" or not resp.results:
                    yield "event: done\ndata: {}\n\n"
                    return

                try:
                    backend = resolve_backend(req.generate)
                except GenerationUnavailable as e:
                    yield f"event: notice\ndata: {_json.dumps({'notice': str(e)})}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    return

                yield f"event: backend\ndata: {_json.dumps({'backend': backend})}\n\n"
                try:
                    if backend == "local":
                        from . import localmodel

                        message = build_user_message(
                            req.prompt, resp.results,
                            definitions=resp.definitions, related=resp.related,
                        )
                        for piece in localmodel.generate_stream(SYSTEM_PROMPT, message):
                            yield f"event: token\ndata: {_json.dumps({'t': piece})}\n\n"
                    else:
                        answer, _ = generate_answer(
                            req.prompt, resp.results, backend=backend,
                            definitions=resp.definitions, related=resp.related,
                        )
                        yield f"event: token\ndata: {_json.dumps({'t': answer})}\n\n"
                except Exception as e:  # a failed generation must not kill the results
                    yield f"event: notice\ndata: {_json.dumps({'notice': f'Generation failed: {e}'})}\n\n"
                yield "event: done\ndata: {}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app
