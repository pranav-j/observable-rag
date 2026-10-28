"""What each retrieval stage did, and how long it took.

Retrieval used to discard everything except the final chunk ids: dense and BM25
scores were dropped the moment the rankings were fused, the RRF score was
dropped on sorting, and the cross-encoder score was dropped on truncation. The
pipeline could tell you WHICH chunks came back but nothing about why one beat
another -- and "why did it retrieve that" is the question worth answering about
a RAG system.

So each stage records its rank and score per chunk, and its own wall time. Two
things need this: the demo's trace panel, which shows a reader what the
retriever actually did, and per-stage latency, which was previously one
`hybrid_search` span covering embed, dense search, BM25 and fusion together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ChunkTrace:
    """One chunk's journey through the stages that considered it."""
    chunk_id: str
    source: str
    url: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    fused_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    final_rank: Optional[int] = None

    @property
    def found_by(self) -> str:
        """Which retriever(s) surfaced it -- the point of fusing two."""
        if self.dense_rank is not None and self.bm25_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.bm25_rank is not None:
            return "bm25"
        return "none"


@dataclass
class RetrievalTrace:
    query: str
    stages_ms: dict[str, float] = field(default_factory=dict)
    chunks: list[ChunkTrace] = field(default_factory=list)
    n_dense: int = 0
    n_bm25: int = 0
    n_fused: int = 0
    n_reranked: int = 0

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages_ms.values()), 3)

    def returned(self) -> list[ChunkTrace]:
        """Only the chunks that made it to the generator, in final order."""
        return sorted((c for c in self.chunks if c.final_rank is not None),
                      key=lambda c: c.final_rank)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "stages_ms": {k: round(v, 3) for k, v in self.stages_ms.items()},
            "total_ms": self.total_ms,
            "counts": {"dense": self.n_dense, "bm25": self.n_bm25,
                       "fused": self.n_fused, "reranked": self.n_reranked},
            "chunks": [{**asdict(c), "found_by": c.found_by}
                       for c in sorted(self.chunks,
                                       key=lambda c: (c.final_rank is None,
                                                      c.final_rank if c.final_rank
                                                      is not None else 0,
                                                      c.fused_rank or 0))],
        }
