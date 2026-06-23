"""Smoke tests. RRF fusion is implemented, so it gets a real assertion."""

from observable_rag.retrieve.hybrid import reciprocal_rank_fusion


def test_rrf_merges_and_ranks():
    dense = ["a", "b", "c"]
    lexical = ["b", "d", "a"]
    fused = reciprocal_rank_fusion([dense, lexical])
    # "a" and "b" appear in both lists, so they should outrank "c" and "d".
    assert fused[0] in {"a", "b"}
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
