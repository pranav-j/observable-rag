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

from ..observability.tracing import span
from ..retrieve.hybrid import retrieve as hybrid_retrieve
from .llm import _load_default_llm  # re-exported for scripts/make_golden.py
from .prompt import ABSTAIN_MESSAGE, build_messages

# Any bracketed token is a citation ATTEMPT; only ones matching source#<int>
# AND naming a retrieved chunk count as real. The model invents plausible-looking
# anchors ("[index#starlette-features]") that silently vanished before we started
# counting them, leaving an answer that looks uncited rather than wrongly cited.
_CITE = re.compile(r"\[([^\]\s]+#[^\]\s]+)\]")


@dataclass
class RagOutput:
    answer: str
    retrieved_chunk_ids: list[str]
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float | None = None
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    contradictory: bool = False


def _is_abstention(text: str) -> bool:
    """Did the model decline?

    Matched on the OPENING of the reply, not anywhere inside it. Substring
    matching counted "I can't just say 'The documentation does not cover this.'"
    as an abstention, and requiring the sentence to be the entire reply meant a
    model that correctly declined and then helpfully noted what the docs DO
    cover was recorded as a hallucination. Anchoring to the start keeps the
    check deterministic and auditable while leaving room to be useful.

    The elaboration after the opening is not itself checked here; it still
    carries citations and is still subject to the judge metrics, and whether it
    contradicts the refusal is measured separately by _is_contradictory.
    """
    return text.strip().lower().startswith(ABSTAIN_MESSAGE.lower())


# Roughly a sentence's worth of prose. Below this the tail is a pointer ("the
# sources discuss X"); well above it the model has written an actual answer.
_CONTRADICTION_CHARS = 240


def _is_contradictory(text: str, citations: list[str]) -> bool:
    """Did it decline and then answer anyway?

    Three prompt revisions tried to stop this and each one traded it against
    hallucination: tightening the bar for declining brought the contradiction
    back, loosening it answered half the unanswerable questions. Neither
    false_abstention_rate nor false_answer_rate sees it -- the reply opens with
    a refusal, so it is counted as an abstention no matter what follows.

    So measure it instead of prompting at it. A reply that opens by declining
    and then produces a substantial, cited answer is its own failure: whatever
    the metrics call it, a reader is told "not covered" and then handed the
    thing that supposedly isn't covered.
    """
    stripped = text.strip()
    if not stripped.lower().startswith(ABSTAIN_MESSAGE.lower()):
        return False
    tail = stripped[len(ABSTAIN_MESSAGE):].strip()
    return len(tail) > _CONTRADICTION_CHARS and bool(citations)


def _parse_citations(answer: str, valid: set[str]) -> tuple[list[str], list[str]]:
    """Split the answer's citation attempts into (resolved, fabricated)."""
    good: list[str] = []
    bad: list[str] = []
    for cid in _CITE.findall(answer):
        target = good if cid in valid else bad
        if cid not in target:
            target.append(cid)
    return good, bad


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
            cited, fabricated = _parse_citations(text, set(ids))
            contradictory = _is_contradictory(text, cited)
            out = RagOutput(
                answer=text,
                retrieved_chunk_ids=ids,
                contexts=[c.text for c in chunks],
                abstained=_is_abstention(text),
                latency_ms=(time.perf_counter() - t0) * 1000,
                citations=cited,
                invalid_citations=fabricated,
                contradictory=contradictory,
            )
            sp.set_attribute("output.abstained", out.abstained)
            sp.set_attribute("output.citations", ", ".join(out.citations))
            sp.set_attribute("output.invalid_citations", ", ".join(fabricated))
            sp.set_attribute("output.contradictory", contradictory)
            return out