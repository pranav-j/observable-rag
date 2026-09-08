"""Entry point: fetch the corpus, chunk it, and build both indexes.

    python -m observable_rag.index.build

Env: DOC_LIMIT caps docs (dev/CI), FASTAPI_REF pins the revision, QDRANT_URL points
at a Qdrant server (else embedded on disk). When using a server we wait for it to be
ready first, since in a container the DB may still be starting.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..ingest.chunk import chunk_document
from ..ingest.load import DEFAULT_REF, fetch_corpus
from .lexical import BM25_PATH, BM25Index
from .vector import QDRANT_PATH, VectorIndex

CHUNKS_PATH = Path("data/corpus/chunks.jsonl")


def _wait_for_qdrant(url: str, timeout: float = 90.0, interval: float = 2.0) -> None:
    from qdrant_client import QdrantClient
    deadline = time.time() + timeout
    while True:
        try:
            QdrantClient(url=url).get_collections()
            return
        except Exception:  # noqa: BLE001
            if time.time() > deadline:
                raise
            print(f"  waiting for qdrant at {url} ...")
            time.sleep(interval)


def main() -> int:
    ref = os.getenv("FASTAPI_REF", DEFAULT_REF)
    limit = int(os.getenv("DOC_LIMIT", "0")) or None

    url = os.getenv("QDRANT_URL")
    if url:
        _wait_for_qdrant(url)

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
    print(f"  vector index  -> {url or QDRANT_PATH} (collection: fastapi_docs)")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())