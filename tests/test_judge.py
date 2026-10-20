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


def test_judge_uses_a_second_model_on_the_same_provider(monkeypatch):
    """No second key needed: one provider, two models is enough for independence,
    and it doesn't route the judge onto someone else's free tier."""
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-nano")
    provider, model, independent = llm.resolve_judge()
    assert provider == "openai"
    assert model == "gpt-4o-mini"
    assert independent is True


def test_judge_skips_a_candidate_that_is_the_generator(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")   # the first judge choice
    provider, model, independent = llm.resolve_judge()
    assert (provider, model) == ("openai", "gpt-4.1-mini")
    assert independent is True


def test_judge_falls_back_to_another_provider_when_one_model_is_all_there_is(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GEMINI_MODEL", llm.DEFAULT_GEMINI_MODEL)  # its only candidate
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    provider, _model, independent = llm.resolve_judge()
    assert provider == "openai"
    assert independent is True


def test_judge_falls_back_to_the_generator_and_says_so(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "g")        # the only key we hold
    monkeypatch.setenv("GEMINI_MODEL", llm.DEFAULT_GEMINI_MODEL)
    provider, model, independent = llm.resolve_judge()
    assert (provider, model) == llm.resolve_generation()
    assert independent is False, "a self-grading judge must not claim independence"


def test_openai_is_the_default_provider(monkeypatch):
    """It used to default to gemini, so a missing .env silently routed every
    call to a free tier capped at 20 requests a day."""
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert llm.resolve_generation()[0] == "openai"


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


# --- citation precision ------------------------------------------------------

from observable_rag.generate.judge import _claim_for  # noqa: E402


def test_claim_is_the_sentence_carrying_the_citation():
    answer = ("Background tasks run after the response [a#0]. "
              "Use Depends to declare a dependency [b#1].")
    assert _claim_for(answer, "b#1") == "Use Depends to declare a dependency."
    assert _claim_for(answer, "a#0") == "Background tasks run after the response."
    assert _claim_for(answer, "missing#9") is None


def test_citation_precision_grades_each_citation():
    seen = []

    def complete(messages):
        seen.append(messages[1]["content"])
        return "1" if "Depends" in messages[0]["content"] + messages[1]["content"] else "0"

    j = LLMJudge(complete=complete)
    score = j.citation_precision(
        "Tasks run later [a#0]. Use Depends here [b#1].",
        {"a#0": "unrelated text", "b#1": "Depends declares a dependency"})
    assert score == 0.5          # one citation supported, one not
    assert len(seen) == 2        # graded per citation, not per answer


def test_citation_precision_without_citations_is_none():
    j = LLMJudge(complete=lambda m: "1")
    assert j.citation_precision("No citations here.", {}) is None


def test_parse_score_ignores_numbers_that_are_not_scores():
    assert _parse_score("7/10") is None      # used to read the '1' of '10' as 1.0
    assert _parse_score("95") is None
    assert _parse_score("0.9/1.0") == 0.9


# --- rate-limit handling -----------------------------------------------------

def test_a_transient_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("observable_rag.generate.judge.time.sleep", lambda _s: None)
    attempts = []

    def flaky(_messages):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("Error code: 429 - rate limit reached, try again in 1.5s")
        return "0.8"

    j = LLMJudge(complete=flaky)
    assert j.faithfulness("q", "a", ["ctx"]) == 0.8
    assert j.retried == 2
    assert j.failed == 0


def test_a_persistent_rate_limit_is_recorded_not_hidden(monkeypatch):
    monkeypatch.setattr("observable_rag.generate.judge.time.sleep", lambda _s: None)

    def dead(_messages):
        raise RuntimeError("Error code: 429 - quota exceeded, retry in 19h57m")

    j = LLMJudge(complete=dead, retries=3)
    assert j.faithfulness("q", "a", ["ctx"]) is None
    assert j.failed == 1
    assert "429" in j.first_error


def test_a_non_rate_limit_error_is_not_retried(monkeypatch):
    calls = []

    def boom(_messages):
        calls.append(1)
        raise RuntimeError("model not found")

    j = LLMJudge(complete=boom)
    assert j.faithfulness("q", "a", ["ctx"]) is None
    assert len(calls) == 1, "only rate limits should be retried"


def test_retry_delay_honours_the_server_but_caps_it():
    from observable_rag.generate.judge import _retry_after
    assert _retry_after("try again in 1.5525s", default=9) == 2.0525
    assert _retry_after("try again in 600s", default=9) == 30.0        # capped
    assert _retry_after("no hint here", default=9) == 9
    # a daily quota reports hours in a format we deliberately don't parse, so it
    # falls back to the short default and gives up quickly instead of sleeping
    assert _retry_after("Please retry in 19h57m52.8s", default=9) == 9


def test_empty_judge_env_vars_do_not_pin_anything(monkeypatch):
    """CI passes `${{ vars.JUDGE_MODEL }}`, which is "" when the variable isn't
    set. That must not be read as "pin the judge to the empty model"."""
    _clear(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-nano")
    monkeypatch.setenv("JUDGE_PROVIDER", "")
    monkeypatch.setenv("JUDGE_MODEL", "")
    assert llm.resolve_judge() == ("openai", "gpt-4o-mini", True)
