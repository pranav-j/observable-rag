"""Phase 3: the generation pipeline the eval harness scores.

retrieve() supplies the reranked chunks; we show them to the LLM tagged with
their ids, force inline [id] citations and an exact abstain phrase, then parse
the answer. Grounding is enforced by construction (only these sources, cite
them, abstain otherwise) -- imperfect on its own, which is why the faithfulness
metric and tracing exist.

The LLM client is injectable and provider-agnostic: _load_default_llm() picks
OpenAI or Gemini from the LLM_PROVIDER env var (default "gemini"). Gemini is
reached via its OpenAI-compatible endpoint, so one chat-completions call serves
both and no extra dependency is needed. Retrieval, reranking, and embeddings are
local, so on Gemini's free Flash tier the whole system runs at zero cost.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from ..ingest.chunk import Chunk
from ..retrieve.hybrid import retrieve as hybrid_retrieve
from .prompt import ABSTAIN_MESSAGE, build_messages

OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_CITE = re.compile(r"\[([^\]\s]+#\d+)\]")


@dataclass
class RagOutput:
    answer: str
    retrieved_chunk_ids: list[str]
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float | None = None
    citations: list[str] = field(default_factory=list)


def _parse_citations(answer: str, valid: set[str]) -> list[str]:
    seen: list[str] = []
    for cid in _CITE.findall(answer):
        if cid in valid and cid not in seen:
            seen.append(cid)
    return seen


def _chat_completer(client, model: str):
    """Wrap an OpenAI-style client into complete(messages) -> str (temperature 0)."""
    def complete(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0)
        return resp.choices[0].message.content or ""
    return complete


def _load_openai_llm():
    from openai import OpenAI
    return _chat_completer(OpenAI(), OPENAI_MODEL)


def _load_gemini_llm():
    from openai import OpenAI  # Gemini via its OpenAI-compatible endpoint
    client = OpenAI(api_key=os.environ.get("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
    return _chat_completer(client, GEMINI_MODEL)


def _load_default_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return _load_gemini_llm()
    if provider == "openai":
        return _load_openai_llm()
    raise ValueError(f"unknown LLM_PROVIDER: {provider!r} (use 'gemini' or 'openai')")


class RagPipeline:
    def __init__(self, complete=None, retrieve=None, top_n: int = 5):
        self._complete = complete
        self._retrieve = retrieve or hybrid_retrieve
        self.top_n = top_n

    @property
    def complete(self):
        if self._complete is None:
            self._complete = _load_default_llm()
        return self._complete

    def answer(self, question: str) -> RagOutput:
        t0 = time.perf_counter()
        chunks = self._retrieve(question, top_n=self.top_n)
        if not chunks:
            return RagOutput(
                answer=ABSTAIN_MESSAGE, retrieved_chunk_ids=[], contexts=[],
                abstained=True, latency_ms=(time.perf_counter() - t0) * 1000, citations=[])
        text = self.complete(build_messages(question, chunks))
        ids = [c.id for c in chunks]
        return RagOutput(
            answer=text,
            retrieved_chunk_ids=ids,
            contexts=[c.text for c in chunks],
            abstained=ABSTAIN_MESSAGE.lower() in text.lower(),
            latency_ms=(time.perf_counter() - t0) * 1000,
            citations=_parse_citations(text, set(ids)),
        )