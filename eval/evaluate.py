"""Offline evaluation harness for the domain RAG system.

Reads a golden set of (question, reference answer, relevant chunk ids) triples,
runs each question through the live RAG pipeline, and scores retrieval and
generation quality. Writes a JSON report that ci_gate.py compares against a
committed baseline for relative-regression gating.

Golden set format (one JSON object per line, .jsonl):

    {"id": "q001", "question": "How do I set a default response status code?",
     "answer": "Pass status_code to the path operation decorator.",
     "relevant_chunk_ids": ["response-status-code#0", "response-status-code#1"],
     "answerable": true, "qtype": "factual", "verified_by": "human"}

    {"id": "q057", "question": "What is FastAPI's built-in rate limiter called?",
     "answer": null, "relevant_chunk_ids": [], "answerable": false,
     "qtype": "unanswerable_in_domain"}

Unanswerable items have answerable=false and answer=null; the correct behaviour
is for the system to abstain rather than invent an answer. They are how you
measure hallucination, and most eval sets omit them entirely.

Metric definitions (see docs/evaluation.md for the longer version):

  recall@k            fraction of an item's relevant chunks that appear in the
                      top k. With one labelled chunk this is hit-or-miss; with
                      several (multi_hop) it is partial credit.
  all_relevant@k      stricter: 1.0 only when EVERY relevant chunk is in the top
                      k. This is the one that means "could answer it at all".
  context_precision@k fraction of the top k that is labelled relevant. Bounded
                      above by |relevant|/k, so it reads low by construction.
  MRR                 1/rank of the FIRST relevant chunk.

Retrieval metrics are computed from what retrieval actually returned, whether or
not the generator went on to abstain. Folding an abstention into recall as a
zero conflates two different failures and hides which stage broke; abstention
has its own metric (false_abstention_rate).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Optional, Protocol, Sequence

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260907


@dataclass
class GoldenItem:
    id: str
    question: str
    answer: Optional[str]
    relevant_chunk_ids: list[str]
    answerable: bool
    qtype: str = "factual"
    verified_by: Optional[str] = None


@dataclass
class RagOutput:
    """What your pipeline must return for each question."""
    answer: str
    retrieved_chunk_ids: list[str]
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False
    latency_ms: Optional[float] = None
    citations: list[str] = field(default_factory=list)


class RagSystem(Protocol):
    """Plug your pipeline in here: hybrid retrieve -> rerank -> generate."""
    def answer(self, question: str) -> RagOutput: ...


class Judge(Protocol):
    """LLM-as-judge metrics. Wire RAGAS (or your own) behind this interface.

    Keeping it pluggable means the deterministic retrieval metrics never depend
    on a fast-moving judge library, and you can pin or swap the judge without
    touching the harness. citation_precision is optional: a judge that doesn't
    implement it simply doesn't contribute that metric.
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
            verified_by=raw.get("verified_by"),
        ))
    return items


# --- deterministic retrieval metrics ---------------------------------------

def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str],
                k: int) -> Optional[float]:
    if not relevant:
        return None
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(relevant)


def all_relevant_at_k(relevant: Sequence[str], retrieved: Sequence[str],
                      k: int) -> Optional[float]:
    """1.0 only if every relevant chunk is in the top k. Matters for multi_hop."""
    if not relevant:
        return None
    return float(set(relevant) <= set(retrieved[:k]))


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


# --- aggregation ------------------------------------------------------------

def _clean(values: Sequence[Optional[float]]) -> list[float]:
    return [v for v in values if v is not None and not math.isnan(v)]


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = _clean(values)
    return round(fmean(clean), 4) if clean else None


def _rate(flags: Sequence[Optional[bool]]) -> Optional[float]:
    clean = [f for f in flags if f is not None]
    return round(sum(1 for f in clean if f) / len(clean), 4) if clean else None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return round(s[idx], 2)


def bootstrap_ci(values: Sequence[Optional[float]], confidence: float = 0.95,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> Optional[list[float]]:
    """Percentile bootstrap interval for the mean.

    With ~26 items a single question is worth ~4 points of recall, so a bare
    point estimate invites reading noise as progress. Seeded, so the same report
    produces the same interval.
    """
    clean = _clean(values)
    if len(clean) < 2:
        return None
    rng = random.Random(seed)
    n = len(clean)
    means = sorted(fmean(rng.choices(clean, k=n)) for _ in range(resamples))
    lo = means[int((1 - confidence) / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 + confidence) / 2 * resamples))]
    return [round(lo, 4), round(hi, 4)]


def _spread(values: Sequence[Optional[float]]) -> Optional[dict]:
    """mean / min / max / stdev across repeated judge runs."""
    clean = _clean(values)
    if not clean:
        return None
    return {
        "mean": round(fmean(clean), 4),
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "stdev": round(pstdev(clean), 4) if len(clean) > 1 else 0.0,
        "runs": len(clean),
    }


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# --- per-item results -------------------------------------------------------

@dataclass
class ItemResult:
    id: str
    qtype: str
    answerable: bool
    question: str
    answer: str
    abstained: bool
    retrieved_chunk_ids: list[str]
    contexts: list[str]
    citations: list[str]
    latency_ms: Optional[float]
    recall: Optional[float] = None
    all_relevant: Optional[float] = None
    precision: Optional[float] = None
    rr: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    citation_precision: Optional[float] = None
    false_answer: Optional[bool] = None
    false_abstention: Optional[bool] = None


def _score_retrieval(item: GoldenItem, out: RagOutput, k: int) -> dict:
    return {
        "recall": recall_at_k(item.relevant_chunk_ids, out.retrieved_chunk_ids, k),
        "all_relevant": all_relevant_at_k(item.relevant_chunk_ids,
                                          out.retrieved_chunk_ids, k),
        "precision": context_precision_at_k(item.relevant_chunk_ids,
                                            out.retrieved_chunk_ids, k),
        "rr": reciprocal_rank(item.relevant_chunk_ids, out.retrieved_chunk_ids),
    }


def _judge_item(judge: Judge, question: str, res: "ItemResult",
                chunk_texts: dict[str, str]) -> dict:
    """Grade one answer.

    Abstentions are skipped: an abstention asserts nothing, so "are its claims
    supported" has no answer, and scoring it either 0 or 1 would be inventing
    one. Whether abstaining was right is what false_answer_rate and
    false_abstention_rate measure. The report records judge_graded counts so the
    size of this exclusion is visible rather than implied.
    """
    scores: dict = {"faithfulness": None, "answer_relevance": None,
                    "citation_precision": None}
    if res.abstained:
        return scores

    scores["faithfulness"] = judge.faithfulness(question, res.answer, res.contexts)
    scores["answer_relevance"] = judge.answer_relevance(question, res.answer,
                                                        res.contexts)
    grade_citations = getattr(judge, "citation_precision", None)
    if grade_citations and res.citations:
        cited = {c: chunk_texts[c] for c in res.citations if chunk_texts.get(c)}
        if cited:
            scores["citation_precision"] = grade_citations(res.answer, cited)
    return scores


# --- evaluation loop --------------------------------------------------------

def run_eval(system: RagSystem, items: list[GoldenItem], k: int = 5,
             judge: Optional[Judge] = None, judge_runs: int = 1,
             chunk_texts: Optional[dict[str, str]] = None,
             metadata: Optional[dict] = None) -> tuple[dict, list[ItemResult]]:
    """Run every golden item once, then grade the answers `judge_runs` times.

    The system is run once per item and the judge replays over those stored
    answers, so repeated runs measure JUDGE noise rather than re-rolling the
    generator too.
    """
    judge = judge or NullJudge()
    chunk_texts = chunk_texts or {}

    results: list[ItemResult] = []
    for it in items:
        out = system.answer(it.question)
        res = ItemResult(
            id=it.id, qtype=it.qtype, answerable=it.answerable,
            question=it.question, answer=out.answer, abstained=out.abstained,
            retrieved_chunk_ids=list(out.retrieved_chunk_ids),
            contexts=list(out.contexts),
            citations=list(getattr(out, "citations", [])),
            latency_ms=out.latency_ms,
        )
        if it.answerable:
            for name, value in _score_retrieval(it, out, k).items():
                setattr(res, name, value)
            res.false_abstention = out.abstained
        else:
            res.false_answer = not out.abstained
        results.append(res)

    # judge pass(es): same answers, graded repeatedly, to measure judge noise
    per_run: list[dict] = []
    samples: dict[str, list[list[Optional[float]]]] = {
        "faithfulness": [], "answer_relevance": [], "citation_precision": []}
    gradable = [(it, r) for it, r in zip(items, results)
                if r.answer.strip() and not r.abstained]
    runs = 0 if isinstance(judge, NullJudge) else max(1, judge_runs)

    for _ in range(runs):
        run_scores: dict[str, list] = {name: [] for name in samples}
        for it, res in gradable:
            for name, value in _judge_item(judge, it.question, res,
                                           chunk_texts).items():
                run_scores[name].append(value)
        for name, values in run_scores.items():
            samples[name].append(values)
        per_run.append({name: _mean(values) for name, values in run_scores.items()})

    # collapse repeats into one value per item
    for name, runs in samples.items():
        if not runs:
            continue
        for idx, (_, res) in enumerate(gradable):
            setattr(res, name, _mean([run[idx] for run in runs]))

    report = _aggregate(results, k=k, judge_runs=per_run, metadata=metadata or {})
    return report, results


def _metric_columns(results: Sequence[ItemResult]) -> dict:
    return {
        "recall": [r.recall for r in results],
        "all_relevant": [r.all_relevant for r in results],
        "precision": [r.precision for r in results],
        "rr": [r.rr for r in results],
        "faithfulness": [r.faithfulness for r in results],
        "answer_relevance": [r.answer_relevance for r in results],
        "citation_precision": [r.citation_precision for r in results],
    }


def _slice(results: Sequence[ItemResult], k: int) -> dict:
    """Every metric for one slice of items (all of them, or one category)."""
    cols = _metric_columns(results)
    return {
        f"recall_at_{k}": _mean(cols["recall"]),
        f"all_relevant_at_{k}": _mean(cols["all_relevant"]),
        f"context_precision_at_{k}": _mean(cols["precision"]),
        "mrr": _mean(cols["rr"]),
        "faithfulness": _mean(cols["faithfulness"]),
        "answer_relevance": _mean(cols["answer_relevance"]),
        "citation_precision": _mean(cols["citation_precision"]),
        "false_answer_rate": _rate([r.false_answer for r in results]),
        "false_abstention_rate": _rate([r.false_abstention for r in results]),
        "n": len(results),
    }


def _aggregate(results: list[ItemResult], k: int, judge_runs: list[dict],
               metadata: dict) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    cols = _metric_columns(results)
    latency = [r.latency_ms for r in results if r.latency_ms is not None]

    graded = {name: len(_clean(cols[name]))
              for name in ("faithfulness", "answer_relevance", "citation_precision")}

    by_category: dict[str, dict] = {}
    for qtype in sorted({r.qtype for r in results}):
        by_category[qtype] = _slice([r for r in results if r.qtype == qtype], k)

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "k": k,
            "n_items": len(results),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "judge_graded": graded,
            **metadata,
        },
        "retrieval": {
            f"recall_at_{k}": _mean(cols["recall"]),
            f"all_relevant_at_{k}": _mean(cols["all_relevant"]),
            f"context_precision_at_{k}": _mean(cols["precision"]),
            "mrr": _mean(cols["rr"]),
        },
        "generation": {
            "faithfulness": _mean(cols["faithfulness"]),
            "answer_relevance": _mean(cols["answer_relevance"]),
            "citation_precision": _mean(cols["citation_precision"]),
        },
        "robustness": {
            "false_answer_rate": _rate([r.false_answer for r in unanswerable]),
            "false_abstention_rate": _rate([r.false_abstention for r in answerable]),
        },
        "latency_ms": {
            "p50": _percentile(latency, 50),
            "p95": _percentile(latency, 95),
        },
        "confidence_intervals": {
            f"recall_at_{k}": bootstrap_ci(cols["recall"]),
            f"all_relevant_at_{k}": bootstrap_ci(cols["all_relevant"]),
            "mrr": bootstrap_ci(cols["rr"]),
            "faithfulness": bootstrap_ci(cols["faithfulness"]),
            "answer_relevance": bootstrap_ci(cols["answer_relevance"]),
            "citation_precision": bootstrap_ci(cols["citation_precision"]),
            "false_answer_rate": bootstrap_ci(
                [float(r.false_answer) for r in unanswerable
                 if r.false_answer is not None]),
            "false_abstention_rate": bootstrap_ci(
                [float(r.false_abstention) for r in answerable
                 if r.false_abstention is not None]),
        },
        "judge_spread": {
            name: _spread([run[name] for run in judge_runs])
            for name in ("faithfulness", "answer_relevance", "citation_precision")
        } if judge_runs else {},
        "per_category": by_category,
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


def _build_system() -> tuple[RagSystem, str]:
    try:
        from observable_rag.generate.answer import RagPipeline  # type: ignore
        return RagPipeline(), "RagPipeline"
    except Exception as e:  # noqa: BLE001
        print(f"! using stub system ({e}) — wire observable_rag.generate.answer."
              f"RagPipeline for real results")
        return _StubSystem(), "stub"


def _build_judge():
    try:
        from observable_rag.generate.judge import LLMJudge
        return LLMJudge()
    except Exception as e:  # noqa: BLE001
        print(f"! judge unavailable ({e}); skipping LLM-graded metrics")
        return None


def _provenance(system_name: str, judge_enabled: bool, judge_runs: int) -> dict:
    """What produced these numbers. A baseline without this can't be compared
    against anything: a model swap looks exactly like a quality change."""
    meta: dict = {"system": system_name}
    try:
        from observable_rag.generate.llm import resolve_generation, resolve_judge
        provider, model = resolve_generation()
        meta["generator"] = {"provider": provider, "model": model}
        if judge_enabled:
            jp, jm, independent = resolve_judge()
            meta["judge"] = {"provider": jp, "model": jm,
                             "independent": independent, "runs": judge_runs}
            if not independent:
                print("! the judge is the same model as the generator; "
                      "faithfulness is self-graded")
        else:
            meta["judge"] = None
    except Exception as e:  # noqa: BLE001
        meta["generator"] = {"error": str(e)}

    manifest_path = Path("data/eval/corpus_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["corpus"] = {"fastapi_ref": manifest.get("fastapi_ref"),
                          "n_chunks": manifest.get("n_chunks")}
    return meta


def _load_chunk_texts() -> dict[str, str]:
    try:
        from observable_rag.ingest.manifest import load_chunk_texts
        return load_chunk_texts()
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the RAG eval harness over the golden set.")
    ap.add_argument("--golden", type=Path, default=Path("data/eval/golden_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--items-out", type=Path, default=Path("eval_items.jsonl"),
                    help="per-item results, used by the calibration sampler")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge-runs", type=int, default=1,
                    help="grade each answer N times to measure judge noise")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM-graded faithfulness / answer-relevance metrics")
    ap.add_argument("--trace", action="store_true",
                    help="export spans to Phoenix (run `phoenix serve` first)")
    args = ap.parse_args()
    if args.trace:
        from observable_rag.observability.tracing import init_tracing
        init_tracing()

    items = load_golden(args.golden)
    judge = None if args.no_judge else _build_judge()
    system, system_name = _build_system()
    metadata = _provenance(system_name, judge is not None, args.judge_runs)

    report, results = run_eval(system, items, k=args.k, judge=judge,
                               judge_runs=args.judge_runs,
                               chunk_texts=_load_chunk_texts(), metadata=metadata)

    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.items_out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nper-item results -> {args.items_out}")


if __name__ == "__main__":
    main()
