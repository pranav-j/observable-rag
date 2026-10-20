"""Tests for the LLM-as-judge metrics, using a fake LLM (no key, no network)."""

import pytest

import observable_rag.generate.llm as llm
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

# --- judge/generator independence -------------------------------------------


def _clear(monkeypatch):
    for var in ("LLM_PROVIDER", "JUDGE_PROVIDER", "JUDGE_MODEL",
                "OPENAI_API_KEY", "OPENAI_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
                "GROQ_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(var, raising=False)


def test_judge_prefers_a_provider_the_generator_is_not_using(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    provider, model, independent = llm.resolve_judge()
    assert provider == "openai"
    assert model == llm.DEFAULT_OPENAI_MODEL
    assert independent is True


def test_judge_falls_back_to_the_generator_and_says_so(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g")        # the only key we hold
    provider, model, independent = llm.resolve_judge()
    assert (provider, model) == llm.resolve_generation()
    assert independent is False, "a self-grading judge must not claim independence"


def test_judge_env_overrides_win(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("JUDGE_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_MODEL", "some-judge-model")
    assert llm.resolve_judge() == ("groq", "some-judge-model", True)


def test_unknown_judge_provider_is_rejected(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("JUDGE_PROVIDER", "nope")
    with pytest.raises(ValueError):
        llm.resolve_judge()
