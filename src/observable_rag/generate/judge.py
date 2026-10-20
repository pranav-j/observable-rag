"""LLM-as-judge metrics for the eval harness: faithfulness and answer relevance.

These are graded by an LLM, so they are noisy run-to-run -- which is exactly why
the CI gate compares relative deltas, not absolute thresholds.

The judge deliberately runs on a DIFFERENT model from the generator (see
llm.resolve_judge): a model grading its own output rates itself generously, and
faithfulness is the metric most sensitive to that. Set JUDGE_PROVIDER/JUDGE_MODEL
to control it; when no second provider is credentialed the judge falls back to
the generator's model and the eval report records independent_judge: false.

The judge degrades to None on any error: a judge failure must never crash the
eval or silently score it zero. The CI gate treats a missing gated metric as a
failure, so that None can't pass as "skip".

    faithfulness       -- are the answer's claims supported by the retrieved sources?
    answer_relevance   -- does the answer actually address the question?
    citation_precision -- does each [chunk] cited actually back the sentence citing it?

citation_precision is the one that catches a specific, common failure: an answer
that is broadly faithful to the retrieved set as a whole, but hangs its citations
on the wrong chunks. Graded per citation, not per answer.
"""

from __future__ import annotations

import re
from collections import Counter
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

_CITATION_PROMPT = (
    "You check one citation. Reply 1 if the SOURCE directly supports the CLAIM, and "
    "0 if it does not, even if the claim is true in general. Reply with ONLY 0 or 1."
)

# Anchored so a stray number can't be read as a score: "7/10" used to match the
# '1' of '10' and come back as a perfect 1.0.
_NUM = re.compile(r"(?<![\d.])(?:[01](?:\.\d+)?|\.\d+)(?![\d.])")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITE_MARKER = re.compile(r"\[[^\]\s]+#\d+\]")


def _parse_score(text: str) -> Optional[float]:
    m = _NUM.search(text or "")
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(0))))
    except ValueError:
        return None


def _claim_for(answer: str, chunk_id: str) -> Optional[str]:
    """The sentence that carries [chunk_id], with citation markers stripped.

    Falls back to the whole answer when the citation isn't inside a sentence we
    can isolate, so a one-line answer still gets graded.
    """
    marker = f"[{chunk_id}]"
    if marker not in answer:
        return None
    for sentence in _SENTENCE_SPLIT.split(answer):
        if marker in sentence:
            return _strip_markers(sentence)
    return _strip_markers(answer)


def _strip_markers(text: str) -> Optional[str]:
    text = _CITE_MARKER.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text).strip()
    return text or None


class LLMJudge:
    """Judge conforming to the harness's Judge protocol, backed by an LLM."""

    def __init__(self, complete=None):
        self._complete = complete
        self.calls = 0
        self.failures: "Counter[str]" = Counter()
        self.first_error: Optional[str] = None

    @property
    def complete(self):
        if self._complete is None:
            from .llm import load_judge_llm
            self._complete = load_judge_llm()
        return self._complete

    def _grade(self, system: str, user: str) -> Optional[float]:
        """Degrade to None on error, but COUNT it.

        Returning None quietly is how a judge that was rate-limited into
        uselessness still produced a confident-looking 1.0 over the handful of
        calls that got through. The caller reads these counters and the eval
        report carries them, so a dead judge is visible instead of flattering.
        """
        self.calls += 1
        try:
            return _parse_score(self.complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}]))
        except Exception as e:  # noqa: BLE001
            kind = type(e).__name__
            self.failures[kind] += 1
            if self.first_error is None:
                self.first_error = f"{kind}: {str(e)[:300]}"
            return None

    @property
    def failed(self) -> int:
        return sum(self.failures.values())

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

    def citation_precision(self, answer: str,
                           cited_chunks: dict[str, str]) -> Optional[float]:
        """Fraction of the answer's citations whose chunk backs the claim it's on.

        Each citation is graded against the sentence it appears in, not the whole
        answer: the failure we're looking for is a right-ish answer wearing the
        wrong citation, and whole-answer grading hides it.
        """
        if not answer.strip() or not cited_chunks:
            return None
        verdicts = []
        for chunk_id, text in cited_chunks.items():
            claim = _claim_for(answer, chunk_id)
            if not claim:
                continue
            score = self._grade(_CITATION_PROMPT,
                                f"SOURCE:\n{text}\n\nCLAIM:\n{claim}")
            if score is not None:
                verdicts.append(1.0 if score >= 0.5 else 0.0)
        return sum(verdicts) / len(verdicts) if verdicts else None