# 4. One provider, two models

Date: 2026-10-21

Supersedes the selection rule in
[0001](0001-judge-runs-on-a-different-model.md). The reason for an independent
judge is unchanged; how one gets chosen is not.

## Context

ADR 0001 made the judge prefer a provider the generator wasn't using. The
thinking was that a different vendor gives stronger independence than a
different model from the same family.

In practice that rule picked whichever provider happened to have a key, and the
extra keys were free-tier accounts left over from building this without paying
for anything. The limits make them unusable for evaluation:

| provider | free limit | what one eval run needs |
| --- | --- | --- |
| Gemini | 20 requests/day | ~190 judge calls |
| Groq | 200k tokens/day | ~190k tokens for a single 3-pass run |

Both ran out mid-run. The first attempt graded 5 of 20 answers and reported
faithfulness 1.0 from the survivors. The second exhausted a daily token cap and
graded almost nothing. Groq's configured default model had also been retired
and returned 404.

Meanwhile the generator runs on a paid OpenAI account with many models
available. The judge requirement is a different *model*; a different provider is
one way to get one, not the requirement itself.

## Decision

Resolve the judge in this order:

1. `JUDGE_PROVIDER` / `JUDGE_MODEL`, if set.
2. **A second model on the generator's own provider**, from a per-provider
   `judge_models` preference list, skipping any entry equal to the generator's
   model.
3. Another provider we hold a key for.
4. The generator's own model, reported as `independent=False`.

Also: the default provider changes from `gemini` to `openai`, so a missing
`.env` no longer routes everything to a capped free tier, and Groq's retired
default model is replaced.

With only `OPENAI_API_KEY` set, the generator is `gpt-4o-mini` and the judge is
`gpt-4.1-mini`. With `OPENAI_MODEL=gpt-5-nano`, the judge is `gpt-4o-mini`.
Either way independence holds with one key and one bill.

## Alternatives

- **Keep preferring another provider.** Strongest independence on paper, and it
  is what produced two unusable runs. A judge that stops answering halfway is
  worse than one sharing a vendor.
- **Pay for a second provider.** Real independence, a second account and bill
  for a portfolio project. Available via `JUDGE_PROVIDER` for anyone who wants
  it.
- **Drop Gemini and Groq entirely.** Simpler, and it throws away a genuine
  feature: the pipeline really is provider-agnostic, and the free tiers are fine
  for trying it out, just not for evaluation.
- **Detect quota and fail over at runtime.** Guessing another provider's
  remaining quota from error strings, mid-eval, with the metric's denominator
  shifting underneath. The coverage guard already catches the outcome.

## Consequences

- One API key is enough for the whole system, generator and judge.
- Judge and generator now usually share a vendor, so correlated biases are
  possible in a way two vendors would avoid. This buys reliability at some cost
  to independence, and it is the weaker form of ADR 0001's guarantee. Judge
  calibration against human labels is what actually settles whether the judge is
  trustworthy; `JUDGE_PROVIDER` remains available when a second vendor matters
  more than reliability.
- `judge_models` is a hard-coded list per provider and will rot as models are
  retired — exactly how Groq's default came to 404. `JUDGE_MODEL` is the escape
  hatch, and a 404 is loud rather than silent.
- Free tiers stay usable for trying the system out, and are now documented as
  unsuitable for an eval run rather than quietly selected for one.
