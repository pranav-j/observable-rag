"""Phase 2: BM25 lexical index for exact identifiers and keywords."""

from __future__ import annotations

from ..ingest.chunk import Chunk


def build_index(chunks: list[Chunk]) -> None:
    raise NotImplementedError("phase 2: build BM25 index")


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, score)] by BM25. TODO: implement."""
    raise NotImplementedError("phase 2: BM25 search")
