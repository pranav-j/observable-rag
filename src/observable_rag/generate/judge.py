"""LLM-as-judge metrics for the eval harness: faithfulness and answer relevance.

These are graded by an LLM, so they are noisy run-to-run -- which is exactly why
the CI gate compares relative deltas, not absolute thresholds. The judge reuses
the provider-agnostic client (so it respects LLM_PROVIDER) and degrades to None on
any error: a judge failure must never crash the eval or silently score it zero.

    faithfulness     -- are the answer's claims supported by the retrieved sources?
    answer_relevance -- does the answer actually address the question?
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_FAITHFULNESS_PROMPT = (
    "You grade whether an ANSWER is supported by the given SOURCES. Score 1.0 if "
    "every claim in the answer is directly supported by the sources, 0.0 if it "
    "contradicts them or adds unsupported claims, and a value in between for partial "
    "support. Reply with ONLY the number between 0 and 1."
)
_RELEVANCE_PROMPT = (
    "You grade whether an ANSWER addresses the QUESTION. Judge only relevance / "
    "on-topic-ness, not factual correctness. Score 1.0 if it directly and completely "
    "addresses the question, 0.0 if off-topic, in between if partial. Reply with ONLY "
    "the number between 0 and 1."
)

_NUM = re.compile(r"[01](?:\.\d+)?|0?\.\d+")


def _parse_score(text: str) -> Optional[float]:
    m = _NUM.search(text or "")
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(0))))
    except ValueError:
        return None


class LLMJudge:
    """Judge conforming to the harness's Judge protocol, backed by an LLM."""

    def __init__(self, complete=None):
        self._complete = complete

    @property
    def complete(self):
        if self._complete is None:
            from .llm import _load_default_llm
            self._complete = _load_default_llm()
        return self._complete

    def _grade(self, system: str, user: str) -> Optional[float]:
        try:
            return _parse_score(self.complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}]))
        except Exception:  # noqa: BLE001
            return None

    def faithfulness(self, question: str, answer: str,
                     contexts: Sequence[str]) -> Optional[float]:
        if not answer.strip() or not contexts:
            return None
        sources = "\n\n".join(contexts)
        return self._grade(_FAITHFULNESS_PROMPT,
                           f"SOURCES:\n{sources}\n\nANSWER:\n{answer}")

    def answer_relevance(self, question: str, answer: str,
                         contexts: Sequence[str]) -> Optional[float]:
        if not answer.strip():
            return None
        return self._grade(_RELEVANCE_PROMPT,
                           f"QUESTION:\n{question}\n\nANSWER:\n{answer}")