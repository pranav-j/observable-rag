"""Phase 2: hybrid retrieval -- fuse dense + BM25 with RRF, then rerank.

Two stages: hybrid_search casts a wide, recall-oriented net (dense + lexical,
combined by RANK not score), then retrieve() rescores those candidates with the
cross-encoder and keeps the best few. Phase 4 wraps each stage in a span.
"""

from __future__ import annotations

from ..ingest.chunk import Chunk
from ..index import lexical, vector
from ..observability.tracing import record_retrieval, span
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
    with span("retrieve", **{"input.value": query, "retrieve.top_n": top_n}) as sp:
        with span("hybrid_search"):
            ids = hybrid_search(query, top_k_each=top_k_each, rrf_k=rrf_k)
        store = load_chunks() if store is None else store
        candidates = [store[cid] for cid in ids if cid in store]
        with span("rerank", **{"rerank.candidates": len(candidates)}):
            result = (reranker or CrossEncoderReranker()).rerank(query, candidates, top_n)
        record_retrieval(sp, query, [c.id for c in result])
        return result