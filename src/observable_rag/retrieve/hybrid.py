"""Phase 2: hybrid retrieval -- fuse dense + BM25 with RRF, then rerank.

Two stages: hybrid_search casts a wide, recall-oriented net (dense + lexical,
combined by RANK not score), then the cross-encoder reranker does the precise,
expensive reordering on just those candidates. Recall first, precision second.

reciprocal_rank_fusion + hybrid_search are already implemented -- read them; RRF
fuses by rank because dense (cosine) and BM25 scores live on incomparable scales.
Your job is retrieve(): wire the two stages together.
"""

from __future__ import annotations

from ..ingest.chunk import Chunk
from ..index import lexical, vector
from .rerank import CrossEncoderReranker
from .store import load_chunks


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(query: str, top_k_each: int = 20, rrf_k: int = 60) -> list[str]:
    dense = [cid for cid, _ in vector.search(query, top_k_each)]
    lexical_hits = [cid for cid, _ in lexical.search(query, top_k_each)]
    return reciprocal_rank_fusion([dense, lexical_hits], k=rrf_k)


def retrieve(query: str, top_k_each: int = 20, rrf_k: int = 60, top_n: int = 5,
             store: "dict[str, Chunk] | None" = None,
             reranker: "CrossEncoderReranker | None" = None) -> list[Chunk]:
    """YOUR IMPLEMENTATION -- the two-stage pipeline:
      1. recall:  ids = hybrid_search(query, top_k_each, rrf_k)
      2. resolve: turn ids into Chunks using `store` (default load_chunks())
      3. precision: rerank with `reranker` (default CrossEncoderReranker()); return top_n
    """
    raise NotImplementedError("wire hybrid_search -> resolve via store -> rerank -> top_n")