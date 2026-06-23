"""Offline evaluation harness for the domain RAG system.

Reads a golden set of (question, reference answer, relevant chunk ids) triples,
runs each question through the live RAG pipeline, and scores retrieval and
generation quality. Writes a JSON report that ci_gate.py compares against a
committed baseline for relative-regression gating.

Golden set format (one JSON object per line, .jsonl):

    {"id": "q001", "question": "How do I set a default response status code?",
     "answer": "Pass status_code to the path operation decorator.",
     "relevant_chunk_ids": ["response-status-code#0", "response-status-code#1"],
     "answerable": true, "qtype": "factual"}

    {"id": "q057", "question": "What is FastAPI's built-in rate limiter called?",
     "answer": null, "relevant_chunk_ids": [], "answerable": false,
     "qtype": "unanswerable"}

Unanswerable items have answerable=false and answer=null; the correct behaviour
is for the system to abstain rather than invent an answer. They are how you
measure hallucination, and most eval sets omit them entirely.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Optional, Protocol, Sequence


@dataclass
class GoldenItem:
    id: str
    question: str
    answer: Optional[str]
    relevant_chunk_ids: list[str]
    answerable: bool
    qtype: str = "factual"


@dataclass
class RagOutput:
    """What your pipeline must return for each question."""
    answer: str
    retrieved_chunk_ids: list[str]
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: Optional[float] = None


class RagSystem(Protocol):
    """Plug your pipeline in here: hybrid retrieve -> rerank -> generate."""
    def answer(self, question: str) -> RagOutput: ...


class Judge(Protocol):
    """LLM-as-judge metrics. Wire RAGAS (or your own) behind this interface.

    Keeping it pluggable means the deterministic retrieval metrics never depend
    on a fast-moving judge library, and you can pin or swap the judge without
    touching the harness.
    """
    def faithfulness(self, question: str, answer: str,
                     contexts: Sequence[str]) -> Optional[float]: ...
    def answer_relevance(self, question: str, answer: str,
                         contexts: Sequence[str]) -> Optional[float]: ...


class NullJudge:
    """Default judge: skips LLM-graded metrics (returns None)."""
    def faithfulness(self, question, answer, contexts): return None
    def answer_relevance(self, question, answer, contexts): return None


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        items.append(GoldenItem(
            id=raw["id"],
            question=raw["question"],
            answer=raw.get("answer"),
            relevant_chunk_ids=raw.get("relevant_chunk_ids", []),
            answerable=raw.get("answerable", raw.get("answer") is not None),
            qtype=raw.get("qtype", "factual"),
        ))
    return items


# --- deterministic retrieval metrics ---------------------------------------

def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str],
                k: int) -> Optional[float]:
    if not relevant:
        return None
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(relevant)


def context_precision_at_k(relevant: Sequence[str], retrieved: Sequence[str],
                           k: int) -> Optional[float]:
    top = retrieved[:k]
    if not top:
        return None
    rel = set(relevant)
    return sum(1 for c in top if c in rel) / len(top)


def reciprocal_rank(relevant: Sequence[str],
                    retrieved: Sequence[str]) -> Optional[float]:
    if not relevant:
        return None
    rel = set(relevant)
    for i, c in enumerate(retrieved, start=1):
        if c in rel:
            return 1.0 / i
    return 0.0


def _mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(fmean(clean), 4) if clean else None


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[idx], 2)


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# --- evaluation loop --------------------------------------------------------

def run_eval(system: RagSystem, items: list[GoldenItem], k: int = 5,
             judge: Optional[Judge] = None) -> dict:
    judge = judge or NullJudge()
    answerable = [it for it in items if it.answerable]
    unanswerable = [it for it in items if not it.answerable]

    recall, precision, rr = [], [], []
    faith, relevance, latency = [], [], []
    wrongly_abstained = 0

    for it in answerable:
        out = system.answer(it.question)
        if out.latency_ms is not None:
            latency.append(out.latency_ms)
        if out.abstained:
            # an abstention on an answerable question is a miss
            wrongly_abstained += 1
            recall.append(0.0)
            precision.append(0.0)
            rr.append(0.0)
            continue
        recall.append(recall_at_k(it.relevant_chunk_ids, out.retrieved_chunk_ids, k))
        precision.append(context_precision_at_k(it.relevant_chunk_ids, out.retrieved_chunk_ids, k))
        rr.append(reciprocal_rank(it.relevant_chunk_ids, out.retrieved_chunk_ids))
        faith.append(judge.faithfulness(it.question, out.answer, out.contexts))
        relevance.append(judge.answer_relevance(it.question, out.answer, out.contexts))

    # hallucination check: did it answer when it should have abstained?
    false_answers = 0
    for it in unanswerable:
        out = system.answer(it.question)
        if out.latency_ms is not None:
            latency.append(out.latency_ms)
        if not out.abstained:
            false_answers += 1

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "k": k,
            "n_items": len(items),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
        },
        "retrieval": {
            f"recall_at_{k}": _mean(recall),
            f"context_precision_at_{k}": _mean(precision),
            "mrr": _mean(rr),
        },
        "generation": {
            "faithfulness": _mean(faith),
            "answer_relevance": _mean(relevance),
        },
        "robustness": {
            "false_answer_rate":
                round(false_answers / len(unanswerable), 4) if unanswerable else None,
            "over_abstain_rate":
                round(wrongly_abstained / len(answerable), 4) if answerable else None,
        },
        "latency_ms": {
            "p50": _percentile(latency, 50),
            "p95": _percentile(latency, 95),
        },
    }


# --- stub so the harness runs before the pipeline exists --------------------

class _StubSystem:
    """Placeholder. Replace with your real pipeline:

        from observable_rag.generate.answer import RagPipeline
        system = RagPipeline(config)
    """
    def answer(self, question: str) -> RagOutput:
        return RagOutput(answer="(stub)", retrieved_chunk_ids=[],
                         abstained=True, latency_ms=1.0)


def _build_system() -> RagSystem:
    try:
        from observable_rag.generate.answer import RagPipeline  # type: ignore
        return RagPipeline()
    except Exception:
        print("! using stub system — wire observable_rag.generate.answer.RagPipeline "
              "for real results")
        return _StubSystem()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the RAG eval harness over the golden set.")
    ap.add_argument("--golden", type=Path, default=Path("data/eval/golden_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    items = load_golden(args.golden)
    report = run_eval(_build_system(), items, k=args.k)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
