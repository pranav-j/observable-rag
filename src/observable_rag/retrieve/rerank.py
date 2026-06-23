"""Phase 2: cross-encoder reranking of fused candidates (local model)."""

from __future__ import annotations


def rerank(query: str, chunk_ids: list[str], top_n: int = 5) -> list[str]:
    """Rescore candidates with a cross-encoder, return top_n ids. TODO: implement."""
    raise NotImplementedError("phase 2: cross-encoder rerank")
