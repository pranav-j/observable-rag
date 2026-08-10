"""Entry point: fetch the corpus, chunk it, and (phase 2) build the indexes.

    python -m observable_rag.index.build

Env: DOC_LIMIT caps how many docs to fetch (handy in CI), FASTAPI_REF pins the
docs revision. Phase 1 fetches and chunks; phase 2 will build the indexes from
the persisted chunks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..ingest.chunk import chunk_document
from ..ingest.load import DEFAULT_REF, fetch_corpus

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
    print("TODO phase 2: embed chunks into Qdrant + build the BM25 index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
