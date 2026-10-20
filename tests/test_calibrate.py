"""Tests for judge calibration: the agreement maths and the stratified sample.

Cohen's kappa is easy to get subtly wrong, and a wrong kappa would be reported
next to the judge metrics as if it validated them.
"""

from calibrate_judge import (
    agreement_report,
    build_sample,
    cohens_kappa,
    interpret_kappa,
    percent_agreement,
    stratified_sample,
)

# --- kappa -------------------------------------------------------------------

def test_perfect_agreement_is_one():
    a = [1, 0, 1, 0, 1, 0]
    assert cohens_kappa(a, a) == 1.0
    assert percent_agreement(a, a) == 1.0


def test_kappa_discounts_agreement_that_chance_explains():
    """Both raters call 90% of answers supported and agree on 80% of them --
    high raw agreement, almost no skill once chance is removed."""
    judge = [1] * 9 + [0]
    human = [1] * 8 + [0, 1]
    assert percent_agreement(judge, human) == 0.8
    assert cohens_kappa(judge, human) < 0.0


def test_kappa_is_zero_when_the_judge_never_varies():
    """A judge that calls everything supported scores 0: perfect agreement on
    the majority class, no skill. Not None -- 0 is the informative answer."""
    assert cohens_kappa([1, 1, 1, 1], [1, 0, 1, 0]) == 0.0
    assert interpret_kappa(0.0) == "none -- no better than chance"


def test_kappa_is_undefined_only_when_both_raters_are_constant():
    assert cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]) is None


def test_kappa_goes_negative_when_raters_oppose():
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0


def test_empty_input_is_none():
    assert cohens_kappa([], []) is None
    assert percent_agreement([], []) is None


# --- report ------------------------------------------------------------------

def _row(rid, qtype, judge_f, human_f, judge_c=None, human_c=None):
    return {"id": rid, "qtype": qtype, "question": "q", "answer": "a",
            "contexts": ["c"], "citations": ["a#0"],
            "judge_faithfulness": judge_f, "judge_citation_precision": judge_c,
            "human_faithful": human_f, "human_citations_correct": human_c}


def test_report_counts_only_labelled_rows():
    rows = [_row("1", "factual", 0.9, True),
            _row("2", "factual", 0.2, False),
            _row("3", "factual", 0.9, None)]      # unlabelled
    report = agreement_report(rows, threshold=0.5)
    assert report["metrics"]["faithfulness"]["n_labelled"] == 2
    assert report["metrics"]["faithfulness"]["percent_agreement"] == 1.0


def test_threshold_changes_the_verdict():
    rows = [_row("1", "factual", 0.7, False), _row("2", "factual", 0.7, False)]
    lenient = agreement_report(rows, threshold=0.5)["metrics"]["faithfulness"]
    strict = agreement_report(rows, threshold=0.8)["metrics"]["faithfulness"]
    assert lenient["percent_agreement"] == 0.0   # judge says supported, human disagrees
    assert strict["percent_agreement"] == 1.0    # judge now says unsupported too


def test_report_handles_a_sample_with_no_labels():
    report = agreement_report([_row("1", "factual", 0.9, None)], threshold=0.5)
    assert report["metrics"]["faithfulness"]["n_labelled"] == 0
    assert report["metrics"]["faithfulness"]["cohens_kappa"] is None


# --- sampling ----------------------------------------------------------------

def _item(rid, qtype, faith, abstained=False):
    return {"id": rid, "qtype": qtype, "question": "q", "answer": "a",
            "abstained": abstained, "faithfulness": faith, "contexts": [],
            "citations": [], "citation_precision": None}


def test_sample_reaches_for_low_scores_first():
    """19 near-perfect answers and one bad one: the bad one must be in a sample
    of 5, or the agreement number is computed from unanimous rows."""
    items = [_item(f"q{i}", "factual", 0.99) for i in range(19)]
    items.append(_item("bad", "factual", 0.1))
    picked = {r["id"] for r in stratified_sample(items, 5)}
    assert "bad" in picked


def test_sample_spreads_across_qtypes():
    items = ([_item(f"f{i}", "factual", 0.99) for i in range(10)] +
             [_item("m1", "multi_hop", 0.99), _item("n1", "near_miss", 0.99)])
    picked = {r["qtype"] for r in stratified_sample(items, 4)}
    assert {"multi_hop", "near_miss"} <= picked


def test_abstentions_are_not_sampled():
    """There is nothing to label on an abstention; the judge didn't grade it."""
    items = [_item("a", "factual", None, abstained=True),
             _item("b", "factual", 0.9)]
    assert [r["id"] for r in stratified_sample(items, 5)] == ["b"]


def test_sample_never_writes_a_human_label():
    rows = build_sample([_item("a", "factual", 0.9)], 1, seed=0)
    assert rows[0]["human_faithful"] is None
    assert rows[0]["human_citations_correct"] is None


def test_sample_is_deterministic():
    items = [_item(f"q{i}", "factual", 0.9) for i in range(20)]
    assert ([r["id"] for r in stratified_sample(items, 6, seed=1)] ==
            [r["id"] for r in stratified_sample(items, 6, seed=1)])
