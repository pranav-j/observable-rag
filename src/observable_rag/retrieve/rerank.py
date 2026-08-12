"""Phase 2b (YOUR IMPLEMENTATION): cross-encoder reranking -- the precision stage.

A cross-encoder scores each (query, chunk) PAIR jointly. That's far more accurate
than the bi-encoder (which encoded query and chunk separately), but it's
O(candidates) expensive, so it runs only on the small fused candidate set -- never
the whole corpus. The scorer is injectable (the default lazily loads the model),
so your logic is testable without downloading anything.

TODO 1 -- _load_default_scorer(): return a callable score(query, texts) -> list[float]
          backed by sentence-transformers CrossEncoder:
              from sentence_transformers import CrossEncoder
              model = CrossEncoder(RERANK_MODEL)
              model.predict([(query, text), ...])   # -> list[float], higher = better

TODO 2 -- CrossEncoderReranker.rerank(): score (query, chunk.text) for every chunk,
          sort by score descending, return the top_n chunks (Chunks, not ids).
          Return [] for empty input.

Make tests/test_rerank.py pass (the rerank tests use a fake scorer -- no model needed).
"""

from __future__ import annotations

from ..ingest.chunk import Chunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_default_scorer():
    raise NotImplementedError("TODO 1: wrap sentence-transformers CrossEncoder")


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
        raise NotImplementedError("TODO 2: score pairs, sort desc, return top_n chunks")