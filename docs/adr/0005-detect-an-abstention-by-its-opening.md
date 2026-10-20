# 5. Detect an abstention by its opening

Date: 2026-10-21

## Context

`false_answer_rate` is the project's headline hallucination metric: of the
questions the docs genuinely don't cover, how many got an answer instead of a
refusal. It depends entirely on classifying a reply as an abstention, which was
done by testing whether the exact sentence "The documentation does not cover
this." appeared *anywhere* in the reply.

Rewriting the generation prompt to fix a separate problem (fabricated citations)
exposed how fragile that is. With a reworded prompt the model started declining
in its own words:

> No—FastAPI docs do not mention a built-in WebSocket pub/sub broker.

> ...there is no built-in cron-style scheduler described in these sources.

Both are correct refusals. Both were recorded as hallucinations, taking
`false_answer_rate` from 0.000 to 0.333 while the system's actual behaviour was
right in all six cases. Tightening the prompt to demand the sentence *and
nothing else* made it worse, not better: the model would rather be helpful than
terse, and it reliably added a sentence explaining what the docs do cover.

So the metric was measuring prompt compliance, not honesty. The original 0.000
was luck — one phrasing of the prompt that happened to produce literal
compliance — not a property of the system.

Substring matching had a second, opposite flaw: a reply discussing the phrase
("I wouldn't say 'The documentation does not cover this.' here") counted as an
abstention.

## Decision

An abstention is a reply whose **opening** is the exact abstain sentence:

```python
text.strip().lower().startswith(ABSTAIN_MESSAGE.lower())
```

and the prompt asks the model to *begin* with that sentence, explicitly allowing
it to add a sentence or two afterwards about what the sources do cover.

This keeps the check deterministic and auditable — no judge, no fuzzy matching,
no second opinion about what counts as a refusal — while letting the model be
useful. Measured over the golden set, `false_answer_rate` returned to 0.000 and
`false_abstention_rate` stayed at 0.000, with fabricated citations also at zero.

## Alternatives

- **Keep demanding the sentence verbatim and nothing else.** Tried. Made it
  worse twice. It fights the model's strong preference for being helpful, and
  every prompt edit silently re-rolls a robustness metric.
- **Match refusals semantically**, with a pattern list or a judge. Would catch
  the paraphrases, and would also start counting hedged non-answers as
  refusals — which is exactly the failure the metric exists to detect. A
  hallucination metric that relies on a judge to recognise hallucination is
  circular.
- **Require a structured field** (JSON with an `abstained` boolean). Robust, and
  it changes the answer format the demo and citations are built around for a
  single metric's benefit.

## Consequences

- `false_answer_rate` and `false_abstention_rate` change definition. Numbers
  from before this are not comparable, which matters because the committed
  baseline predates it and will need regenerating.
- The text after the opening sentence is not checked by this rule. A model could
  decline correctly and then write something unsupported, and this would still
  count as an abstention. That trailing text still carries citations and is
  still subject to the judge metrics, but it is a real gap: the rule verifies
  the decline, not everything that follows it.
- The metric still depends on the prompt. It is now a much weaker dependency —
  "start with this sentence" rather than "emit only this sentence" — but any
  prompt change should be checked against `false_answer_rate` rather than
  assumed neutral.
- Changing the abstain sentence itself remains a breaking change for every
  stored result.
