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


def _fake_searches(monkeypatch, dense, sparse):
    """retrieve() calls the two searches directly now, so it can time and score
    them separately; patching hybrid_search no longer intercepts anything."""
    monkeypatch.setattr(hybrid.vector, "search", lambda q, k: dense)
    monkeypatch.setattr(hybrid.lexical, "search", lambda q, k: sparse)


def test_retrieve_pipeline(monkeypatch):
    _fake_searches(monkeypatch,
                   dense=[("a#0", 0.9), ("b#0", 0.8)],
                   sparse=[("b#0", 4.2), ("c#0", 1.1)])
    store = {c.id: c for c in CHUNKS}
    r = CrossEncoderReranker(score=_word_overlap)
    out = hybrid.retrieve("declare a dependency with Depends",
                          store=store, reranker=r, top_n=2)
    assert [c.id for c in out] == ["b#0", "a#0"]


def test_retrieve_trace_records_every_stage(monkeypatch):
    """The scores used to be dropped at each stage, so the pipeline could say
    which chunks came back but never why one beat another."""
    _fake_searches(monkeypatch,
                   dense=[("a#0", 0.9), ("b#0", 0.8)],
                   sparse=[("b#0", 4.2), ("c#0", 1.1)])
    store = {c.id: c for c in CHUNKS}
    out, trace = hybrid.retrieve("declare a dependency with Depends",
                                 store=store,
                                 reranker=CrossEncoderReranker(score=_word_overlap),
                                 top_n=2, with_trace=True)

    assert [c.id for c in out] == ["b#0", "a#0"]
    assert set(trace.stages_ms) == {"dense_search", "bm25_search", "fusion", "rerank"}
    assert trace.total_ms >= 0

    by_id = {c.chunk_id: c for c in trace.chunks}
    assert by_id["b#0"].found_by == "both"      # the point of fusing two retrievers
    assert by_id["a#0"].found_by == "dense"
    assert by_id["c#0"].found_by == "bm25"

    b = by_id["b#0"]
    assert b.dense_rank == 2 and b.dense_score == 0.8
    assert b.bm25_rank == 1 and b.bm25_score == 4.2
    assert b.rrf_score is not None and b.fused_rank == 1
    assert b.rerank_score is not None and b.final_rank == 1

    # c#0 was retrieved and fused but didn't survive the cut
    assert by_id["c#0"].fused_rank is not None
    assert by_id["c#0"].final_rank is None
    assert [c.chunk_id for c in trace.returned()] == ["b#0", "a#0"]


def test_retrieve_without_trace_still_returns_a_plain_list(monkeypatch):
    _fake_searches(monkeypatch, dense=[("a#0", 0.9)], sparse=[("b#0", 1.0)])
    out = hybrid.retrieve("q", store={c.id: c for c in CHUNKS},
                          reranker=CrossEncoderReranker(score=_word_overlap),
                          top_n=2)
    assert isinstance(out, list)
    assert all(hasattr(c, "text") for c in out)

def test_reranker_and_store_are_reused_across_queries(monkeypatch):
    """They used to be rebuilt per call, and the cross-encoder loads lazily, so
    every query re-read the model from disk."""
    import observable_rag.retrieve.hybrid as hybrid

    monkeypatch.setattr(hybrid, "_reranker", None)
    monkeypatch.setattr(hybrid, "_store", None)

    built = []
    monkeypatch.setattr(hybrid, "CrossEncoderReranker",
                        lambda *a, **k: built.append(1) or object())
    loads = []
    monkeypatch.setattr(hybrid, "load_chunks", lambda: loads.append(1) or {})

    for _ in range(5):
        hybrid.get_reranker()
        hybrid.get_store()

    assert built == [1], "the cross-encoder was constructed more than once"
    assert loads == [1], "chunks.jsonl was re-read more than once"
