"""Phase 2b: cross-encoder reranking -- the precision stage.

The bi-encoder in vector.py encodes query and chunk SEPARATELY (fast, precomputed).
A cross-encoder feeds each (query, chunk) pair through the model TOGETHER -- far
more accurate, but O(candidates), so it runs only on the small fused candidate set.

The scorer is injectable (default lazily loads the model), so the ranking logic is
testable without downloading anything. The model name is env-configurable
(RERANK_MODEL) because model repos get renamed/reformatted -- config absorbs churn.
"""

from __future__ import annotations

import os

from ..ingest.chunk import Chunk

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")


def _load_default_scorer():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(RERANK_MODEL)

    def score(query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in model.predict([(query, t) for t in texts])]

    return score


class CrossEncoderReranker:
    def __init__(self, score=None, model_name: str = RERANK_MODEL):
        self._score = score
        self.model_name = model_name

    @property
    def score(self):
        if self._score is None:
            self._score = _load_default_scorer()
        return self._score

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []
        scores = self.score(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [chunk for chunk, _ in ranked[:top_n]]