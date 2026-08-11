"""Phase 2a tests. BM25 runs for real (tiny, pure-Python). The vector index is
tested with an in-memory Qdrant and a fake embedder, so the Qdrant plumbing is
exercised without downloading the sentence-transformers model."""

import math

from qdrant_client import QdrantClient

from observable_rag.index.lexical import BM25Index
from observable_rag.index.vector import VectorIndex
from observable_rag.ingest.chunk import Chunk


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source=cid.split("#")[0], url="https://example.test/", text=text)


CHUNKS = [
    _chunk("a#0", "You set the response status_code in the path operation decorator."),
    _chunk("b#0", "Use Depends to declare a dependency in FastAPI."),
    _chunk("c#0", "Background tasks run after the response is returned."),
]

_VOCAB = ["status_code", "depends", "background", "response", "dependency"]


def _fake_embed(texts):
    out = []
    for t in texts:
        tl = t.lower()
        v = [float(tl.count(w)) for w in _VOCAB]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def test_bm25_ranks_exact_identifier_first():
    idx = BM25Index().build(CHUNKS)
    assert idx.search("status_code", top_k=3)[0][0] == "a#0"


def test_bm25_matches_depends():
    idx = BM25Index().build(CHUNKS)
    assert idx.search("Depends", top_k=1)[0][0] == "b#0"


def test_bm25_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "bm25.pkl")
    BM25Index().build(CHUNKS).save(p)
    loaded = BM25Index.load(p)
    assert loaded.search("status_code", top_k=1)[0][0] == "a#0"


def test_vector_search_returns_nearest():
    idx = VectorIndex(client=QdrantClient(":memory:"), collection="test", embed=_fake_embed)
    idx.build(CHUNKS)
    assert idx.search("use Depends for a dependency", top_k=3)[0][0] == "b#0"