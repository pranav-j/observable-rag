# CI gate margins

**Status: applied.** These are the margins in `eval/ci_gate.py`, sized against
the 104-question golden set (78 answerable, 26 unanswerable) and the baseline in
`data/eval/baseline.json`.

The gate fails a pull request when a metric moves in the worse direction by more
than its margin, relative to the baseline. What follows is how each number was
chosen and what was wrong with the ones it replaced.

## Applied margins

Every judge and robustness margin is **3x the standard deviation measured
across three full runs of identical code** (`reports/eval_run{1,2,3}.json`),
rounded to the nearest whole question. Retrieval measured a standard deviation
of exactly zero, so those margins are sized for drift instead.

| metric | margin | measured stdev | 3x stdev |
| --- | --- | --- | --- |
| `recall_at_5` | 0.03 | **0.0000** | — (2 questions of 78) |
| `all_relevant_at_5` | 0.03 | **0.0000** | — |
| `context_precision_at_5` | 0.03 | **0.0000** | — |
| `mrr` | 0.03 | **0.0000** | — |
| `faithfulness` | 0.03 | 0.0090 | 0.027 |
| `answer_relevance` | 0.02 | 0.0053 | 0.016 |
| `citation_precision` | 0.10 | 0.0336 | 0.101 |
| `citation_validity` | 0.02 | 0.0032 | 0.010 |
| `false_answer_rate` | 0.08 | 0.0181 | 0.054 → 2 of 26 |
| `false_abstention_rate` | 0.04 | 0.0105 | 0.031 |
| `contradiction_rate` | 0.03 | 0.0060 | 0.018 |
| `latency_ms.p95` | 15000 ms | — | measured spread, see below |

Twelve metrics are gated, up from six. Two things stand out.

**Retrieval is exactly deterministic.** Three runs, identical to four decimal
places on `recall_at_5`, `all_relevant_at_5`, `context_precision_at_5` and
`mrr`. This had been asserted here on reasoning; now it is measured. Their
margins are therefore not absorbing noise — there is none — they are absorbing
corpus drift, and a tight margin is the correct setting.

**`citation_precision` is the noisiest metric by a distance**, at 3.7x the
standard deviation of faithfulness and with a spread of 0.071 across identical
runs. Its previous margin of 0.05 was *tighter than its own noise floor*, so it
would have failed on wobble exactly as the 150 ms latency margin did. Widening
it to 0.10 is the one margin here that got looser, and it is the direct
consequence of measuring rather than guessing.

`false_abstention_rate` matters particularly: without it the gate can be
satisfied by abstaining more often.

## What was wrong with the old margins

They had never been checked against the size of the golden set. With 20
answerable and 6 unanswerable items, one question flipping moved a metric by a
large fixed amount:

| metric | margin | one question was worth | effect |
| --- | --- | --- | --- |
| `recall_at_5` | 0.03 | 0.050 | any single regression failed |
| `faithfulness` | 0.03 | 0.050 | any single regression failed |
| `answer_relevance` | 0.03 | 0.050 | any single regression failed |
| `false_answer_rate` | 0.02 | 0.167 | any change at all failed |
| `latency_ms.p95` | 150 ms | — | 0.49% tolerance on a 30s metric |

The latency gate was the worst of these. Run against a report from identical
code with every other metric flat or better, it failed:

```
latency_ms.p95   30517.960  34805.660  +4287.700  FAIL
```

A 14% swing between two runs of the same system. The gate was red on provider
variance alone.

## What is and isn't noisy

This is what decides whether a margin is absorbing noise or hiding a regression:

- **`recall_at_5`, `all_relevant_at_5`, `context_precision_at_5`, `mrr`,
  `citation_validity`** are deterministic. Given the same index and the same
  code they return the same number every time. Their only run-to-run movement is
  the corpus changing underneath them — which is drift, and the fix is pinning
  the docs revision, not widening a margin. They stay tight on purpose: a margin
  below one-question granularity means "any regression fails", which is correct
  when there is no noise to absorb.
- **`faithfulness`, `answer_relevance`, `citation_precision`** are graded by a
  model, and vary with *regeneration* rather than with the judge itself.
- **`latency_ms`** end-to-end is mostly the provider's latency, which no pull
  request controls.

## How the judge margins stopped being guesses

The original plan was to set them to 3x the standard deviation across repeated
**judge** runs. That run happened — `reports/judge_noise_2026-10-20.json`, three passes
over the same stored answers, `gpt-4o-mini` judging a `gpt-5-nano` generator,
full coverage, no errors:

| metric | mean | min | max | stdev |
| --- | --- | --- | --- | --- |
| `faithfulness` | 0.9500 | 0.9500 | 0.9500 | **0.0000** |
| `answer_relevance` | 1.0000 | 1.0000 | 1.0000 | **0.0000** |
| `citation_precision` | 0.9062 | 0.9062 | 0.9062 | **0.0000** |

Zero variance, three times out of three — because at `temperature=0` over
identical stored answers the judge is deterministic. The three passes are the
same computation performed three times, and 3 × 0 is not a usable margin.

A trap worth naming: an earlier attempt on a rate-limited provider *did* show
spread (`citation_precision` 0.8846–0.9062). That was not judge noise. Different
calls were dropped on each pass, so the metric's denominator moved between runs.
Apparent noise from a degraded run looks exactly like the real thing, which is
why the report now carries `judge_coverage` and `judge_errors`.

So the variance these margins must absorb is in **regeneration**, not judging —
the generator is not deterministic at `temperature=0` even though the judge is.

Three full runs of identical code measured it:

| metric | run 1 | run 2 | run 3 | stdev |
| --- | --- | --- | --- | --- |
| `faithfulness` | 0.9875 | 0.9684 | 0.9683 | 0.0090 |
| `answer_relevance` | 0.9875 | 0.9810 | 0.9939 | 0.0053 |
| `citation_precision` | 0.9201 | 0.8489 | 0.8489 | 0.0336 |
| `citation_validity` | 0.9673 | 0.9737 | 0.9745 | 0.0032 |
| `false_answer_rate` | 0.1923 | 0.1923 | 0.2308 | 0.0181 |
| `false_abstention_rate` | 0.0385 | 0.0513 | 0.0256 | 0.0105 |
| `contradiction_rate` | 0.0256 | 0.0256 | 0.0128 | 0.0060 |

The margins in the table above are 3x these, rounded to a whole question. Both
other runs pass the gate against run 1 as the baseline, and a simulated
regression of four questions losing their chunk plus three more hallucinations
still fails it.

This also sets the bar for believing a change. A prompt revision that moves
`false_answer_rate` by 0.04 has moved it by about two standard deviations of
pure noise, which is not evidence of anything on its own.

## Latency keeps a loose gate rather than none

An earlier draft of this document proposed removing the latency gate entirely.
Four runs since then give a measured spread instead of a guess: p95 between
30.5s and 43.3s on near-identical code, a 12.8s range. A 15s margin absorbs that
while still catching a doubling, which is strictly better than removing the
check and leaving a genuine blow-up undetected.

End-to-end p95 is still mostly the provider's latency. Per-stage retrieval
latency — embed, dense search, BM25, fusion, rerank — is local and deterministic
and is the right thing to gate. Those spans don't exist separately yet;
`hybrid_search` is still a single span.

## `citation_validity`

Added after a run showed the model emitting citation tags that look like
documentation heading anchors (`tutorial/middleware#adding-asgi-middlewares`)
rather than chunk ids (`tutorial/middleware#3`). Parsing discarded them, so an
answer citing two invented chunks was indistinguishable from one citing nothing,
and `citation_precision` never saw them because nothing survived to grade.

It measures the share of citation *attempts* that name a retrieved chunk. On the
run that found it, 0.561 — 18 of 41 attempts fabricated. After the generation
prompt was rewritten to spell out the format, 1.000 on that set and 0.944 on the
current 104 questions. Deterministic and cheap, so it is gated.

## Still outstanding

**Pin the docs revision.** `FASTAPI_REF` still defaults to `master`. Until it
names a tag or commit, the corpus moves under the baseline and the deterministic
retrieval metrics drift for reasons unrelated to any code change. The manifest
check reports it, but these margins assume a fixed corpus.

**Replace the four judge placeholders with a measurement**, from repeated full
runs rather than repeated judge passes.
