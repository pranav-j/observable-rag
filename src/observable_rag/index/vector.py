"""Phase 2: embed chunks and upsert to Qdrant; dense similarity search."""

from __future__ import annotations

from ..ingest.chunk import Chunk


def index_chunks(chunks: list[Chunk]) -> None:
    raise NotImplementedError("phase 2: embed + upsert to Qdrant")


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    """Return [(chunk_id, score)] by dense similarity. TODO: implement."""
    raise NotImplementedError("phase 2: dense vector search")
