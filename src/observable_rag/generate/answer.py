"""Phase 3: the generation pipeline the eval harness scores (Phase 4: traced).

retrieve() supplies the reranked chunks; we show them to the LLM tagged with their
ids, force inline [id] citations and an exact abstain phrase, then parse the answer.
The LLM client lives in llm.py so this module and the judge share it without a
cycle. answer() is wrapped in spans so Phoenix shows retrieve vs generate.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from ..ingest.chunk import Chunk
from ..observability.tracing import span
from ..retrieve.hybrid import retrieve as hybrid_retrieve
from .llm import _load_default_llm  # re-exported for scripts/make_golden.py
from .prompt import ABSTAIN_MESSAGE, build_messages

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
        with span("rag_pipeline", **{"input.value": question}) as sp:
            t0 = time.perf_counter()
            chunks = self._retrieve(question, top_n=self.top_n)
            if not chunks:
                sp.set_attribute("output.abstained", True)
                return RagOutput(
                    answer=ABSTAIN_MESSAGE, retrieved_chunk_ids=[], contexts=[],
                    abstained=True, latency_ms=(time.perf_counter() - t0) * 1000,
                    citations=[])
            with span("generate"):
                text = self.complete(build_messages(question, chunks))
            ids = [c.id for c in chunks]
            out = RagOutput(
                answer=text,
                retrieved_chunk_ids=ids,
                contexts=[c.text for c in chunks],
                abstained=ABSTAIN_MESSAGE.lower() in text.lower(),
                latency_ms=(time.perf_counter() - t0) * 1000,
                citations=_parse_citations(text, set(ids)),
            )
            sp.set_attribute("output.abstained", out.abstained)
            sp.set_attribute("output.citations", ", ".join(out.citations))
            return out