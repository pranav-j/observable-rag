"""Tests for the eval harness itself: the metric definitions and the aggregation.

These had no coverage, which is awkward for the one piece of code whose whole job
is producing the numbers everything else is judged by. A fake system and a fake
judge keep it offline.
"""

import pytest
from evaluate import (
    GoldenItem,
    NullJudge,
    RagOutput,
    all_relevant_at_k,
    bootstrap_ci,
    context_precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_eval,
)

# --- metric definitions ------------------------------------------------------

def test_recall_gives_partial_credit_for_multi_chunk_items():
    assert recall_at_k(["a", "b"], ["a", "x", "y"], 5) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b", "y"], 5) == 1.0
    assert recall_at_k(["a"], ["x", "y"], 5) == 0.0


def test_recall_respects_the_cutoff():
    assert recall_at_k(["a"], ["x", "y", "z", "w", "v", "a"], 5) == 0.0
    assert recall_at_k(["a"], ["x", "y", "z", "w", "a"], 5) == 1.0


def test_all_relevant_is_all_or_nothing():
    assert all_relevant_at_k(["a", "b"], ["a", "x"], 5) == 0.0
    assert all_relevant_at_k(["a", "b"], ["a", "b"], 5) == 1.0


def test_context_precision_is_bounded_by_the_label_count():
    # one labelled chunk, five retrieved -> 0.2 is the ceiling, not a bad score
    assert context_precision_at_k(["a"], ["a", "b", "c", "d", "e"], 5) == 0.2


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["b"], ["a", "b", "c"]) == 0.5
    assert reciprocal_rank(["c"], ["a", "b"]) == 0.0


def test_metrics_are_none_without_labels():
    assert recall_at_k([], ["a"], 5) is None
    assert reciprocal_rank([], ["a"], ) is None


# --- bootstrap ---------------------------------------------------------------

def test_bootstrap_brackets_the_mean():
    values = [1.0] * 18 + [0.0] * 2
    lo, hi = bootstrap_ci(values)
    assert lo < 0.9 < hi


def test_bootstrap_is_deterministic():
    values = [1.0, 0.0, 1.0, 1.0, 0.5]
    assert bootstrap_ci(values) == bootstrap_ci(values)


def test_bootstrap_needs_more_than_one_value():
    assert bootstrap_ci([1.0]) is None
    assert bootstrap_ci([]) is None


# --- the loop ----------------------------------------------------------------

ITEMS = [
    GoldenItem("q1", "answerable one", "ref", ["a#0"], True, "factual"),
    GoldenItem("q2", "answerable two", "ref", ["b#0"], True, "exact_identifier"),
    GoldenItem("q3", "not covered", None, [], False, "unanswerable_in_domain"),
]
ABSTAIN = "The documentation does not cover this."


class _System:
    """Answers q1 well, abstains on q2 (a false abstention), abstains on q3."""
    def answer(self, question):
        if question == "answerable one":
            return RagOutput(answer="Yes [a#0].", retrieved_chunk_ids=["a#0", "z#9"],
                             contexts=["chunk a"], citations=["a#0"], latency_ms=10.0)
        return RagOutput(answer=ABSTAIN, retrieved_chunk_ids=["b#0"],
                         contexts=["chunk b"], abstained=True, latency_ms=20.0)


class _Judge:
    def faithfulness(self, q, a, c): return 0.8
    def answer_relevance(self, q, a, c): return 0.9
    def citation_precision(self, answer, cited): return 1.0


def _run(**kw):
    return run_eval(_System(), ITEMS, k=5, chunk_texts={"a#0": "chunk a"}, **kw)


def test_false_abstention_counts_answerable_items_that_refused():
    report, _ = _run(judge=NullJudge())
    assert report["robustness"]["false_abstention_rate"] == 0.5   # q2 of q1,q2
    assert report["robustness"]["false_answer_rate"] == 0.0       # q3 abstained


def test_retrieval_is_scored_even_when_the_generator_abstained():
    """q2 abstained but retrieval found b#0; that is a generation failure, not a
    retrieval one, and recall should still say so."""
    _, results = _run(judge=NullJudge())
    q2 = next(r for r in results if r.id == "q2")
    assert q2.abstained is True
    assert q2.recall == 1.0


def test_abstentions_are_not_graded_by_the_judge():
    report, results = _run(judge=_Judge())
    q2 = next(r for r in results if r.id == "q2")
    assert q2.faithfulness is None
    assert report["metadata"]["judge_graded"]["faithfulness"] == 1   # only q1


def test_unanswerable_items_do_not_get_retrieval_scores():
    _, results = _run(judge=NullJudge())
    q3 = next(r for r in results if r.id == "q3")
    assert q3.recall is None and q3.rr is None
    assert q3.false_answer is False


def test_per_category_breakdown_covers_every_qtype():
    report, _ = _run(judge=_Judge())
    cats = report["per_category"]
    assert set(cats) == {"factual", "exact_identifier", "unanswerable_in_domain"}
    assert cats["factual"]["n"] == 1
    assert cats["exact_identifier"]["false_abstention_rate"] == 1.0
    assert cats["factual"]["false_abstention_rate"] == 0.0


def test_repeated_judge_runs_report_spread():
    scores = iter([0.6, 0.9] * 20)

    class _Noisy(_Judge):
        def faithfulness(self, q, a, c): return next(scores)

    report, _ = run_eval(_System(), ITEMS, k=5, judge=_Noisy(), judge_runs=2,
                         chunk_texts={"a#0": "chunk a"})
    spread = report["judge_spread"]["faithfulness"]
    assert spread["runs"] == 2
    assert spread["min"] == 0.6 and spread["max"] == 0.9
    assert spread["stdev"] > 0


def test_null_judge_leaves_generation_metrics_missing():
    report, _ = _run(judge=NullJudge())
    assert report["generation"]["faithfulness"] is None
    assert report["judge_spread"] == {}


def test_report_keeps_the_keys_the_gate_reads():
    report, _ = _run(judge=_Judge())
    assert report["retrieval"]["recall_at_5"] is not None
    assert report["retrieval"]["context_precision_at_5"] is not None
    assert report["retrieval"]["mrr"] is not None
    assert report["generation"]["answer_relevance"] is not None
    assert report["latency_ms"]["p95"] is not None


@pytest.mark.parametrize("metric", ["recall_at_5", "mrr", "faithfulness"])
def test_confidence_intervals_are_present_and_ordered(metric):
    report, _ = _run(judge=_Judge())
    ci = report["confidence_intervals"][metric]
    if ci is not None:
        assert ci[0] <= ci[1]


# --- judge coverage ----------------------------------------------------------

class _FlakyJudge:
    """Grades the first `n` calls, then fails the rest — a rate-limited judge."""
    def __init__(self, n):
        self.budget = n
        self.calls = 0
        self.failures = {}
        self.first_error = None

    def _spend(self):
        self.calls += 1
        if self.budget <= 0:
            self.failures["RateLimitError"] = self.failures.get("RateLimitError", 0) + 1
            self.first_error = self.first_error or "RateLimitError: quota"
            return None
        self.budget -= 1
        return 1.0

    @property
    def failed(self):
        return sum(self.failures.values())

    def faithfulness(self, q, a, c): return self._spend()
    def answer_relevance(self, q, a, c): return self._spend()


MANY = [GoldenItem(f"q{i}", f"question {i}", "ref", ["a#0"], True, "factual")
        for i in range(10)]


class _AlwaysAnswers:
    def answer(self, question):
        return RagOutput(answer=f"Answer [a#0] for {question}.",
                         retrieved_chunk_ids=["a#0"], contexts=["chunk a"],
                         citations=["a#0"], latency_ms=5.0)


def test_a_rate_limited_judge_nulls_the_metric_instead_of_averaging_survivors():
    """The real failure this came from: a daily quota ran out after a handful of
    calls and faithfulness came back 1.0 over 5 of 20 items."""
    report, _ = run_eval(_AlwaysAnswers(), MANY, k=5, judge=_FlakyJudge(4),
                         chunk_texts={"a#0": "chunk a"})
    assert report["generation"]["faithfulness"] is None, \
        "a metric covering 20% of items must not be reported as a number"
    assert report["metadata"]["judge_coverage"]["faithfulness"] < 0.8
    assert "faithfulness" in report["metadata"]["judge_undercovered"]


def test_judge_errors_are_recorded_in_the_report():
    report, _ = run_eval(_AlwaysAnswers(), MANY, k=5, judge=_FlakyJudge(4),
                         chunk_texts={"a#0": "chunk a"})
    errors = report["metadata"]["judge_errors"]
    assert errors["failed_calls"] > 0
    assert "RateLimitError" in errors["by_type"]


def test_full_coverage_still_reports_the_metric():
    report, _ = run_eval(_AlwaysAnswers(), MANY, k=5, judge=_FlakyJudge(1000),
                         chunk_texts={"a#0": "chunk a"})
    assert report["generation"]["faithfulness"] == 1.0
    assert report["metadata"]["judge_coverage"]["faithfulness"] == 1.0
    assert report["metadata"]["judge_undercovered"] == []


def test_citation_validity_counts_attempts_not_just_survivors():
    from evaluate import ItemResult, citation_validity

    def item(good, bad):
        return ItemResult(id="x", qtype="factual", answerable=True, question="q",
                          answer="a", abstained=False, retrieved_chunk_ids=[],
                          contexts=[], citations=good, invalid_citations=bad,
                          contradictory=False, latency_ms=1.0)

    assert citation_validity([item(["a#0"], [])]) == 1.0
    assert citation_validity([item([], ["ghost#x"])]) == 0.0
    assert citation_validity([item(["a#0"], ["ghost#x"])]) == 0.5
    assert citation_validity([item([], [])]) is None       # no attempts to judge
