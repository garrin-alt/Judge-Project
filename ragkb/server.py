"""REST API + web UI server for ragkb."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from .chunking import chunk_text
from .embeddings import embed_query, embed_texts
from .generate import GenerationUnavailable, generate_answer
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
            if store.count_chunks() == 0:
                return {"results": [], "answer": None, "backend": None,
                        "notice": "Knowledge base is empty."}
            qvec = embed_query(req.prompt)
            results = store.search(qvec, top_k=req.top_k)

        payload = {
            "results": [
                {
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "source": r.source,
                    "text": r.text,
                    "score": round(r.score, 4),
                }
                for r in results
            ],
            "answer": None,
            "backend": None,
            "notice": None,
        }

        if req.generate != "none" and results:
            try:
                answer, backend = generate_answer(req.prompt, results, backend=req.generate)
                payload["answer"] = answer
                payload["backend"] = backend
            except GenerationUnavailable as e:
                payload["notice"] = str(e)

        return payload

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app
