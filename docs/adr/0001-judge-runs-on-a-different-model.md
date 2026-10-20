# 1. The judge runs on a different model from the generator

Date: 2026-10-20

> The selection rule here was superseded by
> [0004](0004-one-provider-two-models.md): the judge now prefers a second model
> on the generator's own provider rather than a different provider. Why an
> independent judge is needed, below, still stands.

## Context

`faithfulness` and `answer_relevance` are graded by an LLM. Until now the judge
and the generator were not merely the same model — they were the same object:
`LLMJudge.complete` and `RagPipeline.complete` both called `_load_default_llm()`,
so both read `LLM_PROVIDER` and resolved to one model with one set of weights.

The system therefore graded its own output. That biases the result in a
direction that is easy to predict and hard to detect: a model asked whether its
own answer is supported by the sources tends to say yes, because the same
reasoning that produced the answer produces the justification. The committed
baseline reports faithfulness 0.993 and answer relevance 1.0, which are exactly
the numbers a self-grading setup produces whether or not the answers are good.

Nothing in the report recorded which model produced the numbers, so this was
invisible to anyone reading `baseline.json`.

## Decision

The judge resolves its model separately from the generator, in
`llm.resolve_judge()`:

1. `JUDGE_PROVIDER` and `JUDGE_MODEL` win if set.
2. Otherwise pick any provider we hold credentials for that the generator is
   **not** using.
3. If there is no second credentialed provider, fall back to the generator's
   model and return `independent=False`.

The eval report records the generator's provider and model, the judge's provider
and model, and `judge.independent`. When the judge is not independent, the
harness prints a warning and the report says so, so the number is never
presented as an independent check when it isn't.

## Alternatives

- **Prompt the same model harder.** Stricter rubrics and chain-of-thought
  reduce the effect but don't remove it, and we'd have no way to tell how much
  was left.
- **Require a second provider.** Cleaner, but makes the eval unrunnable for
  anyone with one key, including on a fork. The fallback plus an honest flag
  keeps it runnable and keeps the reader informed.
- **Use a human for every judgement.** That's the ground truth, not a
  replacement for an automated metric. It's how we calibrate the judge instead —
  see `scripts/calibrate_judge.py`.
- **Drop the judge metrics.** They would stop misleading, and also stop being
  useful. The failure they catch — a fluent answer unsupported by the retrieved
  text — isn't visible in any deterministic metric.

## Consequences

- Faithfulness and answer relevance are expected to **drop** when the judge
  changes. That is the bias being removed, not a regression, so the baseline
  must be regenerated rather than compared across the change.
- A second API key is now worth having, and the eval costs slightly more:
  judge calls go to a second provider, which may have its own rate limits.
- The judge becomes another moving part. Changing the judge model changes the
  metric, which is why the report records it and why the margins the CI gate
  uses are derived from measured judge noise rather than assumed.
- An independent judge is still not a calibrated one. Agreement against human
  labels is reported separately.
