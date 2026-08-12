"""Resolve chunk ids back to full Chunks (id -> text/url), loaded from chunks.jsonl.

Retrieval fuses ranked ids; reranking and citation both need the underlying text,
so this id -> Chunk lookup is what they share.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ingest.chunk import Chunk

CHUNKS_PATH = "data/corpus/chunks.jsonl"


def load_chunks(path: str = CHUNKS_PATH) -> dict[str, Chunk]:
    store: dict[str, Chunk] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        store[d["id"]] = Chunk(**d)
    return store