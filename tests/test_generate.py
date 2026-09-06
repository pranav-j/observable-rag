"""Phase 3 tests. A fake LLM and fake retriever keep generation testable with no
API key and no index; we assert the wiring: contract, citations, abstention, and
that the provider switch selects the right backend."""

import observable_rag.generate.answer as ans
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
    monkeypatch.setattr(ans, "_load_openai_llm", lambda: "openai-llm")
    monkeypatch.setattr(ans, "_load_gemini_llm", lambda: "gemini-llm")
    monkeypatch.setattr(ans, "_load_groq_llm", lambda: "groq-llm")
    for provider, expected in [("gemini", "gemini-llm"), ("openai", "openai-llm"),
                               ("groq", "groq-llm")]:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert ans._load_default_llm() == expected


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
    complete = ans._chat_completer(client, "temp-locked-model")
    assert complete([{"role": "user", "content": "hi"}]) == "ok"
    assert complete([{"role": "user", "content": "again"}]) == "ok"
    calls = client.chat.completions.calls
    assert len(calls) == 3                      # 1st: temp attempt + retry; 2nd: no temp
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert "temperature" not in calls[2]