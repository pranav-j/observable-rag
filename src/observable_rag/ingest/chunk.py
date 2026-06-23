"""Phase 1: split Documents into Chunks with STABLE ids.

Chunk ids are the contract the golden set is written against, so the scheme must
not change once the golden set exists. Scheme: f"{document.source}#{index}".
"""

from __future__ import annotations

from dataclasses import dataclass

from .load import Document


@dataclass
class Chunk:
    id: str  # f"{source}#{index}" -- STABLE; do not change after the golden set exists
    source: str
    url: str
    text: str


def chunk_document(doc: Document, max_tokens: int = 512,
                   overlap_tokens: int = 64) -> list[Chunk]:
    """Split one Document into ordered, stably-identified Chunks. TODO: implement."""
    raise NotImplementedError("phase 1: implement token-aware chunking")
