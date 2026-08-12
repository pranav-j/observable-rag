"""Phase 2b tests. A fake word-overlap scorer stands in for the cross-encoder,
so reranking and the two-stage pipeline are tested without the model or an index."""

from observable_rag.ingest.chunk import Chunk
from observable_rag.retrieve import hybrid
from observable_rag.retrieve.rerank import CrossEncoderReranker


def _c(cid, text):
    return Chunk(id=cid, source=cid.split("#")[0], url="https://example.test/", text=text)


CHUNKS = [
    _c("a#0", "Background tasks run after the response"),
    _c("b#0", "Use Depends to declare a dependency"),
    _c("c#0", "Set the response status_code"),
]


def _word_overlap(query, texts):
    qs = set(query.lower().split())
    return [float(len(qs & set(t.lower().split()))) for t in texts]


def test_rerank_orders_by_score_and_truncates():
    r = CrossEncoderReranker(score=_word_overlap)
    out = r.rerank("declare a dependency with Depends", CHUNKS, top_n=2)
    assert [c.id for c in out] == ["b#0", "a#0"]
    assert len(out) == 2


def test_rerank_empty():
    r = CrossEncoderReranker(score=_word_overlap)
    assert r.rerank("anything", [], top_n=5) == []


def test_retrieve_pipeline(monkeypatch):
    monkeypatch.setattr(hybrid, "hybrid_search", lambda q, **kw: ["a#0", "b#0", "c#0"])
    store = {c.id: c for c in CHUNKS}
    r = CrossEncoderReranker(score=_word_overlap)
    out = hybrid.retrieve("declare a dependency with Depends",
                          store=store, reranker=r, top_n=2)
    assert [c.id for c in out] == ["b#0", "a#0"]