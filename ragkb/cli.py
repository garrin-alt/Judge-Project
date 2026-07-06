"""Command-line interface for the ragkb local knowledge base."""

import argparse
import sys
from pathlib import Path

from . import config
from .chunking import chunk_text
from .embeddings import embed_query, embed_texts
from .generate import GenerationUnavailable, generate_answer
from .store import KnowledgeStore


def cmd_add(args):
    if args.file:
        path = Path(args.file)
        text = path.read_text(encoding="utf-8")
        title = args.title or path.name
        source = str(path)
    elif args.text:
        text = args.text
        title = args.title or (text[:60] + ("..." if len(text) > 60 else ""))
        source = args.source
    else:
        print("error: provide either --file or --text", file=sys.stderr)
        return 1

    chunks = chunk_text(text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    if not chunks:
        print("error: no content to add (empty text)", file=sys.stderr)
        return 1

    print(f"Embedding {len(chunks)} chunk(s)...")
    vectors = embed_texts(chunks)

    with KnowledgeStore(args.db) as store:
        doc_id = store.add_document(title, source, chunks, vectors)

    print(f"Added document #{doc_id} '{title}' ({len(chunks)} chunk(s)).")
    return 0


def cmd_query(args):
    with KnowledgeStore(args.db) as store:
        if store.count_chunks() == 0:
            print("Knowledge base is empty. Add something first with `ragkb add`.")
            return 0

        query_vec = embed_query(args.prompt)
        results = store.search(query_vec, top_k=args.top_k)

    if not results:
        print("No relevant results found.")
        return 0

    print(f"Top {len(results)} match(es):\n")
    for i, r in enumerate(results, 1):
        snippet = r.text if len(r.text) <= 300 else r.text[:300] + "..."
        print(f"{i}. [{r.score:.3f}] {r.title}\n   {snippet}\n")

    if args.no_generate:
        return 0

    try:
        answer = generate_answer(args.prompt, results, model=args.model)
    except GenerationUnavailable as e:
        print(f"(Answer generation skipped: {e})")
        return 0

    print("Answer:\n" + answer)
    return 0


def cmd_list(args):
    with KnowledgeStore(args.db) as store:
        docs = store.list_documents()

    if not docs:
        print("Knowledge base is empty.")
        return 0

    for doc_id, title, source, created_at, num_chunks in docs:
        src = f" ({source})" if source else ""
        print(f"#{doc_id}\t{title}{src}\t{num_chunks} chunk(s)")
    return 0


def cmd_remove(args):
    with KnowledgeStore(args.db) as store:
        removed = store.remove_document(args.doc_id)
    if removed:
        print(f"Removed document #{args.doc_id}.")
        return 0
    print(f"No document with id #{args.doc_id}.", file=sys.stderr)
    return 1


def cmd_reset(args):
    if not args.yes:
        confirm = input("This will delete the entire knowledge base. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 1
    with KnowledgeStore(args.db) as store:
        store.reset()
    print("Knowledge base cleared.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="ragkb", description="Local RAG knowledge base.")
    parser.add_argument("--db", type=str, default=None, help="Path to the knowledge base database file.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Add a document or text snippet to the knowledge base.")
    p_add.add_argument("--file", type=str, help="Path to a text file to ingest.")
    p_add.add_argument("--text", type=str, help="Raw text to ingest.")
    p_add.add_argument("--title", type=str, help="Title for this entry.")
    p_add.add_argument("--source", type=str, help="Optional source label (for --text entries).")
    p_add.set_defaults(func=cmd_add)

    p_query = sub.add_parser("query", help="Query the knowledge base with a prompt.")
    p_query.add_argument("prompt", type=str, help="The question or prompt to search for.")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    p_query.add_argument("--no-generate", action="store_true", help="Only show retrieved chunks, skip LLM answer.")
    p_query.add_argument("--model", type=str, default=config.ANTHROPIC_MODEL, help="Anthropic model to use for generation.")
    p_query.set_defaults(func=cmd_query)

    p_list = sub.add_parser("list", help="List documents in the knowledge base.")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Remove a document by id.")
    p_remove.add_argument("doc_id", type=int)
    p_remove.set_defaults(func=cmd_remove)

    p_reset = sub.add_parser("reset", help="Clear the entire knowledge base.")
    p_reset.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
