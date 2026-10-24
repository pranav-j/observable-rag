"""Phase 3 tests. A fake LLM and fake retriever keep generation testable with no
API key and no index; we assert the wiring: contract, citations, abstention, and
that the provider switch selects the right backend."""

import observable_rag.generate.llm as llm
from observable_rag.generate.answer import RagPipeline
from observable_rag.generate.prompt import ABSTAIN_MESSAGE
from observable_rag.ingest.chunk import Chunk


def _c(cid, text):
    return Chunk(id=cid, source=cid.split("#")[0], url="https://example.test/", text=text)


CHUNKS = [_c("a#0", "Background tasks run after the response."),
          _c("b#0", "Use Depends to declare a dependency.")]


def _retrieve(q, top_n=5, **kw):
    return CHUNKS[:top_n]


def test_answer_carries_contract_and_citations():
    pipe = RagPipeline(complete=lambda m: "Use Depends [b#0].", retrieve=_retrieve)
    out = pipe.answer("how do I declare a dependency?")
    assert out.retrieved_chunk_ids == ["a#0", "b#0"]
    assert out.contexts == [c.text for c in CHUNKS]
    assert out.citations == ["b#0"]
    assert out.abstained is False
    assert out.latency_ms is not None


def test_bogus_citations_are_dropped():
    pipe = RagPipeline(complete=lambda m: "See [b#0] and [made-up#9].", retrieve=_retrieve)
    assert pipe.answer("?").citations == ["b#0"]


def test_abstains_when_nothing_retrieved():
    called = []
    pipe = RagPipeline(complete=lambda m: called.append(1) or "x",
                       retrieve=lambda q, **kw: [])
    out = pipe.answer("unanswerable?")
    assert out.abstained is True
    assert out.retrieved_chunk_ids == []
    assert called == []


def test_abstains_when_model_emits_abstain_phrase():
    pipe = RagPipeline(complete=lambda m: ABSTAIN_MESSAGE, retrieve=_retrieve)
    assert pipe.answer("?").abstained is True


def test_llm_provider_switch(monkeypatch):
    monkeypatch.setattr(llm, "_load_openai_llm", lambda: "openai-llm")
    monkeypatch.setattr(llm, "_load_gemini_llm", lambda: "gemini-llm")
    monkeypatch.setattr(llm, "_load_groq_llm", lambda: "groq-llm")
    for provider, expected in [("gemini", "gemini-llm"), ("openai", "openai-llm"),
                               ("groq", "groq-llm")]:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert llm._load_default_llm() == expected


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Completions:
    def __init__(self): self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise RuntimeError(
                "Unsupported value: 'temperature' does not support 0 with this model")
        return _Resp("ok")


class _Chat:
    def __init__(self): self.completions = _Completions()


class _Client:
    def __init__(self): self.chat = _Chat()


def test_chat_completer_retries_without_temperature_then_caches():
    client = _Client()
    complete = llm._chat_completer(client, "temp-locked-model")
    assert complete([{"role": "user", "content": "hi"}]) == "ok"
    assert complete([{"role": "user", "content": "again"}]) == "ok"
    calls = client.chat.completions.calls
    assert len(calls) == 3                      # 1st: temp attempt + retry; 2nd: no temp
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert "temperature" not in calls[2]

def test_empty_env_vars_fall_back_to_defaults(monkeypatch):
    """CI passes `${{ vars.X }}`, which is an EMPTY STRING when the variable
    isn't set -- not an absent one. getenv's default never fires for those."""
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("OPENAI_MODEL", "")
    assert llm.resolve_generation() == (llm.DEFAULT_PROVIDER,
                                        llm.PROVIDERS[llm.DEFAULT_PROVIDER]["default_model"])


def test_fabricated_citations_are_reported_not_just_dropped():
    """gpt-5-nano answered 'What two libraries is FastAPI built on?' with
    [index#starlette-features] -- plausible-looking, not a real chunk id. Those
    used to vanish, leaving an answer that looked uncited rather than miscited."""
    pipe = RagPipeline(
        complete=lambda m: "Starlette and Pydantic [a#0][index#starlette-features].",
        retrieve=_retrieve)
    out = pipe.answer("what is it built on?")
    assert out.citations == ["a#0"]
    assert out.invalid_citations == ["index#starlette-features"]


def test_abstention_is_matched_on_the_opening_not_anywhere():
    """Declining and then saying what the docs DO cover is still a decline; a
    sentence that merely mentions the phrase is not."""
    declined = RagPipeline(
        complete=lambda m: f"{ABSTAIN_MESSAGE} The sources cover Depends [a#0].",
        retrieve=_retrieve)
    assert declined.answer("?").abstained is True

    talked_about_it = RagPipeline(
        complete=lambda m: f"I would not say '{ABSTAIN_MESSAGE}' here [a#0].",
        retrieve=_retrieve)
    assert talked_about_it.answer("?").abstained is False

    paraphrased = RagPipeline(
        complete=lambda m: "No, the docs do not mention that [a#0].",
        retrieve=_retrieve)
    assert paraphrased.answer("?").abstained is False, \
        "a paraphrase must not count: the exact opening is the contract"


def test_declining_then_answering_is_flagged_as_contradictory():
    """Three prompt revisions failed to stop this, and neither robustness rate
    sees it: the reply opens with a refusal so it counts as an abstention
    whatever follows. Measured instead of prompted at."""
    answered_anyway = RagPipeline(
        complete=lambda m: (
            f"{ABSTAIN_MESSAGE} However, it does explain that you can declare a "
            "dependency by passing a callable to Depends(), which FastAPI then "
            "resolves for each request and caches within that request's scope, "
            "so the same dependency used twice is only computed once [a#0]. "
            "You can also use it at router level [b#0]."),
        retrieve=_retrieve)
    out = answered_anyway.answer("how do I use Depends?")
    assert out.abstained is True        # still counted as a decline...
    assert out.contradictory is True    # ...but the contradiction is visible


def test_a_clean_decline_with_a_pointer_is_not_contradictory():
    """Declining and naming what the sources DO cover is the intended shape."""
    pipe = RagPipeline(
        complete=lambda m: f"{ABSTAIN_MESSAGE} The sources discuss CORS [a#0].",
        retrieve=_retrieve)
    out = pipe.answer("?")
    assert out.abstained is True
    assert out.contradictory is False


def test_a_plain_answer_is_not_contradictory():
    pipe = RagPipeline(complete=lambda m: "Use Depends [a#0]." * 20, retrieve=_retrieve)
    assert pipe.answer("?").contradictory is False
