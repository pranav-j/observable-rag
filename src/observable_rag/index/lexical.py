"""Phase 2a: BM25 lexical index over chunks.

Bag-of-words ranking that excels at exact terms and identifiers (status_code,
Depends) where dense embeddings blur. Tokenisation keeps \\w+ runs, so
underscored identifiers stay whole. The index is pickled to disk so retrieval
can load it without rebuilding.
"""

from __future__ import annotations

import pickle
import re

from rank_bm25 import BM25Okapi

from ..ingest.chunk import Chunk

BM25_PATH = "data/index/bm25.pkl"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, bm25: "BM25Okapi | None" = None,
                 chunk_ids: list[str] | None = None):
        self.bm25 = bm25
        self.chunk_ids = chunk_ids or []

    def build(self, chunks: list[Chunk]) -> "BM25Index":
        self.chunk_ids = [c.id for c in chunks]
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        return self

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_k]]

    def save(self, path: str = BM25_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path: str = BM25_PATH) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(bm25=data["bm25"], chunk_ids=data["chunk_ids"])


_default: "BM25Index | None" = None


def get_index() -> BM25Index:
    global _default
    if _default is None:
        _default = BM25Index.load()
    return _default


def build_index(chunks: list[Chunk]) -> BM25Index:
    idx = BM25Index().build(chunks)
    idx.save()
    return idx


def search(query: str, top_k: int) -> list[tuple[str, float]]:
    return get_index().search(query, top_k)