# 3. Score retrieval separately from abstention

Date: 2026-10-20

## Context

The harness used to treat an abstention on an answerable question as a total
failure of that item: recall, context precision and reciprocal rank were all
forced to 0.0 and the item was skipped before the judge saw it.

That conflates two different failures. If retrieval returned the right chunk and
the generator refused to use it, recall was not 0 — retrieval did its job and
generation didn't. Recording it as a retrieval miss points the next hour of
debugging at the wrong stage.

It also made the judge metrics an average over a filtered subset. Abstentions
were skipped before grading, and unanswerable items were never graded at all, so
faithfulness was computed only over answerable questions that produced an
answer — the easiest cases. The report gave no indication that this filtering
had happened, or how many items it removed.

## Decision

Retrieval metrics are computed from what retrieval actually returned, whether or
not the generator went on to abstain. Abstention gets its own two metrics:

- `false_abstention_rate` — answerable items that abstained
- `false_answer_rate` — unanswerable items that answered

Judge metrics still skip abstentions, for a different and narrower reason: an
abstention asserts nothing, so "are its claims supported by the sources" has no
answer, and scoring it either 0 or 1 would be inventing one. What changed is
that `metadata.judge_graded` now records how many items each judge metric
actually covered, so the exclusion is visible in the report instead of implied
by the code.

## Alternatives

- **Keep folding abstention into recall.** One number instead of three, at the
  cost of not knowing which stage failed. The whole point of the project is
  being able to see inside a query.
- **Score an abstention's faithfulness as 1.0** ("it claimed nothing, so it
  claimed nothing false"). Technically defensible and actively misleading: a
  system that abstains on everything would report perfect faithfulness.
- **Score it 0.0.** Equally arbitrary in the other direction, and it would make
  faithfulness move whenever the abstention rate moved, which is what the
  robustness metrics are for.

## Consequences

- `recall_at_5` and `mrr` are now measures of retrieval alone. On the current
  committed baseline this changes nothing numerically, because
  `over_abstain_rate` is 0.0 — no answerable item abstained — but the two will
  diverge as soon as one does.
- `over_abstain_rate` is renamed `false_abstention_rate`, which is what it
  measures and pairs symmetrically with `false_answer_rate`. An old baseline
  carrying the previous name simply won't match that key; the metric is not
  gated, so nothing breaks.
- A system can still game the pair in one direction at a time — abstain on
  everything and `false_answer_rate` goes to zero — which is exactly why both
  are reported and why `false_abstention_rate` exists.
