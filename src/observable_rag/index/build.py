"""Entry point: fetch the corpus, chunk it, and build both indexes.

    python -m observable_rag.index.build

Env: DOC_LIMIT caps how many docs to fetch (handy in CI), FASTAPI_REF pins the
docs revision. Writes data/corpus/chunks.jsonl, the BM25 index, and the Qdrant
vector collection. The first run downloads the embedding model (one time).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..ingest.chunk import chunk_document
from ..ingest.load import DEFAULT_REF, fetch_corpus
from .lexical import BM25_PATH, BM25Index
from .vector import QDRANT_PATH, VectorIndex

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")


def main() -> int:
    ref = os.getenv("FASTAPI_REF", DEFAULT_REF)
    limit = int(os.getenv("DOC_LIMIT", "0")) or None

    docs = fetch_corpus(ref=ref, limit=limit)
    chunks = [c for d in docs for c in chunk_document(d)]

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__) + "\n")
    print(f"fetched {len(docs)} docs -> {len(chunks)} chunks -> {CHUNKS_PATH}")

    Path("data/index").mkdir(parents=True, exist_ok=True)
    BM25Index().build(chunks).save(BM25_PATH)
    print(f"  bm25 index    -> {BM25_PATH}")
    VectorIndex().build(chunks)
    print(f"  vector index  -> {QDRANT_PATH} (collection: fastapi_docs)")

    print("done. next: phase 2b (hybrid fusion + rerank)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())