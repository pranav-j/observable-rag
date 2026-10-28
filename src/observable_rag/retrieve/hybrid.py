"""Phase 2: hybrid retrieval -- fuse dense + BM25 with RRF, then rerank.

Two stages: hybrid_search casts a wide, recall-oriented net (dense + lexical,
combined by RANK not score), then retrieve() rescores those candidates with the
cross-encoder and keeps the best few. Phase 4 wraps each stage in a span.
"""

from __future__ import annotations

import time

from ..index import lexical, vector
from ..ingest.chunk import Chunk
from ..observability.tracing import record_retrieval, span
from .rerank import CrossEncoderReranker
from .store import load_chunks
from .trace import ChunkTrace, RetrievalTrace


def _rrf_scores(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fused (chunk_id, score) pairs, best first. The score is what makes the
    ordering explainable; reciprocal_rank_fusion below drops it."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    return [cid for cid, _ in _rrf_scores(rankings, k)]


def hybrid_search(query: str, top_k_each: int = 20, rrf_k: int = 60) -> list[str]:
    dense = [cid for cid, _ in vector.search(query, top_k_each)]
    lexical_hits = [cid for cid, _ in lexical.search(query, top_k_each)]
    return reciprocal_rank_fusion([dense, lexical_hits], k=rrf_k)


_reranker: "CrossEncoderReranker | None" = None
_store: "dict[str, Chunk] | None" = None


def get_reranker() -> CrossEncoderReranker:
    """One cross-encoder for the process.

    This used to be constructed per call, and the model loads lazily on first
    use, so every single query re-read the cross-encoder from disk -- seconds of
    the measured latency, on a stage that should cost milliseconds. Same pattern
    as index.vector.get_index() and index.lexical.get_index().
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def get_store() -> "dict[str, Chunk]":
    """chunks.jsonl parsed once, not re-read on every query."""
    global _store
    if _store is None:
        _store = load_chunks()
    return _store


def retrieve(query: str, top_k_each: int = 20, rrf_k: int = 60, top_n: int = 5,
             store: "dict[str, Chunk] | None" = None,
             reranker: "CrossEncoderReranker | None" = None,
             with_trace: bool = False):
    """Hybrid retrieve then rerank.

    Returns the chunks. With with_trace=True returns (chunks, RetrievalTrace),
    which carries each stage's wall time and every candidate's rank and score at
    each stage. The default stays a plain list so existing callers are unchanged.

    The stages are timed and spanned separately. They used to share one
    `hybrid_search` span covering embed, dense search, BM25 and fusion, which
    made per-stage latency unmeasurable -- the reranker turned out to be 97% of
    retrieval time, and that was invisible.
    """
    store = get_store() if store is None else store
    trace = RetrievalTrace(query=query)
    seen: dict[str, ChunkTrace] = {}

    def track(cid: str) -> "ChunkTrace | None":
        if cid not in seen:
            chunk = store.get(cid)
            if chunk is None:
                return None
            seen[cid] = ChunkTrace(chunk_id=cid, source=chunk.source, url=chunk.url)
        return seen[cid]

    with span("retrieve", **{"input.value": query, "retrieve.top_n": top_n}) as sp:
        with span("dense_search"):
            t = time.perf_counter()
            dense = vector.search(query, top_k_each)
            trace.stages_ms["dense_search"] = (time.perf_counter() - t) * 1000
        for rank, (cid, score) in enumerate(dense, start=1):
            if (c := track(cid)) is not None:
                c.dense_rank, c.dense_score = rank, round(float(score), 4)

        with span("bm25_search"):
            t = time.perf_counter()
            sparse = lexical.search(query, top_k_each)
            trace.stages_ms["bm25_search"] = (time.perf_counter() - t) * 1000
        for rank, (cid, score) in enumerate(sparse, start=1):
            if (c := track(cid)) is not None:
                c.bm25_rank, c.bm25_score = rank, round(float(score), 4)

        with span("fusion"):
            t = time.perf_counter()
            fused = _rrf_scores([[cid for cid, _ in dense], [cid for cid, _ in sparse]],
                                k=rrf_k)
            trace.stages_ms["fusion"] = (time.perf_counter() - t) * 1000
        ids = [cid for cid, _ in fused]
        for rank, (cid, score) in enumerate(fused, start=1):
            if (c := track(cid)) is not None:
                c.rrf_score, c.fused_rank = round(score, 6), rank

        candidates = [store[cid] for cid in ids if cid in store]
        with span("rerank", **{"rerank.candidates": len(candidates)}):
            t = time.perf_counter()
            scored = (reranker or get_reranker()).rerank_scored(query, candidates, top_n)
            trace.stages_ms["rerank"] = (time.perf_counter() - t) * 1000

        result = [chunk for chunk, _ in scored]
        for rank, (chunk, score) in enumerate(scored, start=1):
            if (c := track(chunk.id)) is not None:
                c.rerank_score, c.final_rank = round(float(score), 4), rank

        trace.chunks = list(seen.values())
        trace.n_dense, trace.n_bm25 = len(dense), len(sparse)
        trace.n_fused, trace.n_reranked = len(ids), len(candidates)

        record_retrieval(sp, query, [c.id for c in result],
                         scores=[s for _, s in scored])
        for stage, ms in trace.stages_ms.items():
            sp.set_attribute(f"retrieval.{stage}_ms", round(ms, 3))

        return (result, trace) if with_trace else result