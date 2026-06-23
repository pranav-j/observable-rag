"""Phase 2: fuse dense + BM25 results with reciprocal rank fusion (RRF)."""

from __future__ import annotations

from ..index import lexical, vector


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge several ranked id lists into one combined ranking via RRF."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(query: str, top_k_each: int = 20, rrf_k: int = 60) -> list[str]:
    dense = [cid for cid, _ in vector.search(query, top_k_each)]
    lexical_hits = [cid for cid, _ in lexical.search(query, top_k_each)]
    return reciprocal_rank_fusion([dense, lexical_hits], k=rrf_k)
