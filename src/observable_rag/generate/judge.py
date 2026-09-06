"""Tests for the LLM-as-judge metrics, using a fake LLM (no key, no network)."""

from observable_rag.generate.judge import LLMJudge, _parse_score


def test_parse_score_reads_common_formats():
    assert _parse_score("0.8") == 0.8
    assert _parse_score("Score: 1") == 1.0
    assert _parse_score("0") == 0.0
    assert _parse_score(".75") == 0.75
    assert _parse_score("clearly relevant") is None
    assert _parse_score("1.4") == 1.0  # clamped into range


def test_judge_uses_llm_and_returns_scores():
    j = LLMJudge(complete=lambda m: "0.9")
    assert j.faithfulness("q", "a", ["ctx"]) == 0.9
    assert j.answer_relevance("q", "a", ["ctx"]) == 0.9


def test_judge_degrades_to_none_on_error():
    def boom(_m):
        raise RuntimeError("api down")
    j = LLMJudge(complete=boom)
    assert j.faithfulness("q", "a", ["ctx"]) is None
    assert j.answer_relevance("q", "a", ["ctx"]) is None


def test_faithfulness_needs_context_and_answer():
    j = LLMJudge(complete=lambda m: "1.0")
    assert j.faithfulness("q", "a", []) is None          # no sources to grade against
    assert j.faithfulness("q", "   ", ["ctx"]) is None   # empty answer