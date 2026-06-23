"""Phase 3: the end-to-end pipeline the eval harness scores.

RagPipeline.answer(question) -> RagOutput is the single contract evaluate.py
depends on. This dataclass mirrors eval/evaluate.py:RagOutput.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RagOutput:
    answer: str
    retrieved_chunk_ids: list[str]
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: float | None = None


class RagPipeline:
    """hybrid retrieve -> rerank -> generate (with citations / abstention)."""

    def answer(self, question: str) -> RagOutput:
        # TODO phases 2-3: hybrid_search -> rerank -> generate, time the call, and
        #   set abstained=True when the model returns the abstain message.
        # Scaffold placeholder: abstains so the harness runs end to end today.
        return RagOutput(
            answer="(pipeline not implemented yet)",
            retrieved_chunk_ids=[],
            abstained=True,
            latency_ms=0.0,
        )
