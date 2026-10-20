# CI gate margins: proposal

**Status: proposed, not applied.** `eval/ci_gate.py` still carries the original
margins. Nothing here is live until it's reviewed.

The gate fails a pull request when a metric moves in the worse direction by more
than its margin, relative to `data/eval/baseline.json`. The margins have never
been checked against the size of the golden set or against measured run-to-run
noise, and two of them are currently unusable.

## What's wrong now

With 20 answerable and 6 unanswerable items, one item flipping moves a metric by
a fixed, large amount. Against the committed baseline:

| metric | margin | baseline | one item is worth | effect |
| --- | --- | --- | --- | --- |
| `recall_at_5` | 0.03 | 0.900 | 0.050 | any single item regressing fails |
| `context_precision_at_5` | 0.03 | 0.190 | 0.010 | absorbs one item |
| `faithfulness` | 0.03 | 0.993 | 0.050 | any single item regressing fails |
| `answer_relevance` | 0.03 | 1.000 | 0.050 | any single item regressing fails |
| `false_answer_rate` | 0.02 | 0.167 | 0.167 | any single item regressing fails |
| `latency_ms.p95` | 150 ms | 30518 ms | — | 0.49% tolerance |

Two specific problems:

**The latency gate is broken.** A 150 ms margin on a p95 of 30.5 seconds is a
0.49% tolerance on a number dominated by someone else's API. Running the
committed gate against the last local report — identical code, every other
metric flat or better — fails:

```
latency_ms.p95   30517.960  34805.660  +4287.700  FAIL
```

That is a 14% swing between two runs of the same system. The gate as committed
is red on provider variance alone.

**`false_answer_rate` has a six-item denominator.** It can only take the values
0, 0.167, 0.333 … so a 0.02 margin means any change at all fails. The committed
baseline is already 0.167 — one of the six unanswerable questions got answered.

## What is and isn't noisy

Worth separating, because it changes what a margin is *for*:

- **`recall_at_5`, `all_relevant_at_5`, `context_precision_at_5`, `mrr`** are
  deterministic. Given the same index and the same code they return the same
  number every time. Their only source of run-to-run variation is the corpus
  changing underneath them — which is drift, not noise, and the fix is pinning
  the docs revision, not widening a margin.
- **`faithfulness`, `answer_relevance`, `citation_precision`** are graded by a
  model and genuinely vary run to run. Their margins should come from measuring
  that variation.
- **`latency_ms`** end-to-end is mostly provider latency, which no pull request
  controls.

## Proposed margins

| metric | current | proposed | reasoning |
| --- | --- | --- | --- |
| `recall_at_5` | 0.03 | **0.03** | deterministic; keep it tight. Requires the docs revision to be pinned, or corpus drift trips it |
| `all_relevant_at_5` | — | **0.03** | add; the metric that matters for multi-hop |
| `context_precision_at_5` | 0.03 | **0.03** | unchanged |
| `mrr` | ungated | **0.05** | add; it's a README headline and nothing currently guards it |
| `faithfulness` | 0.03 | **0.05, provisionally** | judge replay is deterministic, so repeats measure nothing — see below |
| `answer_relevance` | 0.03 | **0.05, provisionally** | same |
| `citation_precision` | — | **0.05, provisionally** | same |
| `citation_validity` | — | **0.05** | add; deterministic, and currently 0.561 |
| `false_answer_rate` | 0.02 | **0.10** | with ~16 unanswerable items one flip is 0.0625; 0.10 absorbs one, fails on two |
| `false_abstention_rate` | ungated | **0.10** | add; otherwise the gate can be passed by abstaining more |
| `latency_ms.p95` | 150 ms | **ungate** | replace with retrieval-stage latency once per-stage timings exist |

Two notes on that table.

**The retrieval margins stay tight on purpose.** A margin below the one-item
granularity means "any regression fails", and for a deterministic metric that is
the correct setting — there is no noise for it to absorb. It only becomes a
problem when the corpus moves under the baseline, which is a separate fix.

**The judge margins are provisional, because repeating the judge measured
nothing.** The plan was to take 3× the standard deviation across repeated judge
runs. That run happened — `reports/judge_noise_2026-10-20.json`, three passes
over the same stored answers with `gpt-4o-mini` judging a `gpt-5-nano`
generator, full coverage, no errors:

| metric | mean | min | max | stdev |
| --- | --- | --- | --- | --- |
| `faithfulness` | 0.9500 | 0.9500 | 0.9500 | **0.0000** |
| `answer_relevance` | 1.0000 | 1.0000 | 1.0000 | **0.0000** |
| `citation_precision` | 0.9062 | 0.9062 | 0.9062 | **0.0000** |

Zero variance, three times out of three. That isn't a stable metric — it's a
deterministic one. The judge runs at `temperature=0` over identical stored
answers, so the three passes are the same computation performed three times.
3 × 0 = 0 is not a usable margin.

A trap worth naming: an earlier attempt on a rate-limited provider *did* show
spread (`citation_precision` 0.8846–0.9062). That was not judge noise. Different
calls were dropped on each pass, so the metric's denominator changed between
runs. Apparent noise from a degraded run is easy to mistake for the real thing,
which is why the report now carries `judge_coverage` and `judge_errors`.

So the variance a margin needs to absorb isn't in the judge at all — it's in
**regeneration**. Measuring it means N full runs (generate *and* judge), not N
judge passes over one set of answers, at roughly $0.05 a run. Until that
happens, 0.05 is a deliberate placeholder: wide enough to absorb one item
flipping on a 20-item denominator, narrow enough to catch a collapse. It is not
a measured number and shouldn't be described as one.

**Ungating latency is not the same as ignoring it.** End-to-end p95 is worth
reporting and watching; it just isn't something a pull request can be
responsible for when the LLM call is 95% of it. Per-stage retrieval latency —
embed, dense search, BM25, fusion, rerank — is local, deterministic, and worth
gating. Those spans don't exist separately yet.

## A new candidate: `citation_validity`

The same run turned up a failure nothing was measuring. `citation_validity` is
the share of the model's citation *attempts* that name a chunk actually
retrieved. On the current golden set (`reports/citations_2026-10-20.json`):

**0.561 — 23 of 41 citation attempts resolved; 18 were fabricated**, across 6 of
the 20 answered questions.

The invented ones are all the same shape: Markdown heading anchors rather than
chunk indices — `tutorial/middleware#adding-asgi-middlewares` where the scheme
is `tutorial/middleware#3`. They were being discarded during parsing, so an
answer citing two invented chunks looked identical to one citing nothing, and
`citation_precision` never saw them because there was nothing left to grade.
That is why `citation_precision` reads 0.906 while barely half the citations the
model emitted were real.

It's deterministic and it's cheap, so it belongs in the gate. Note that fixing
the underlying prompt would move it sharply, which is a change to the system
being evaluated and needs its own baseline.

## Dependencies

Two things have to land before these can be applied honestly:

1. **Pin the docs revision.** Until `FASTAPI_REF` names a tag or commit, the
   corpus changes under the baseline and the deterministic metrics drift for
   reasons unrelated to any code change.
2. **Measure judge noise.** One `--judge-runs 3` run on the current golden set
   fills in the three blank margins.

Both also want the larger golden set, which moves every denominator in the first
table and is the single biggest improvement available to the gate.
