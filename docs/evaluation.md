# How the evaluation works

What each number in `eval_report.json` means, and the choices behind it. The
harness is `eval/evaluate.py`; the gate that compares a run to the committed
baseline is `eval/ci_gate.py`.

## The golden set

One JSON object per line in `data/eval/golden_set.jsonl`:

```json
{"id": "q001",
 "question": "How do I set a default response status code?",
 "answer": "Pass status_code to the path operation decorator.",
 "relevant_chunk_ids": ["tutorial/response-status-code#0"],
 "answerable": true,
 "qtype": "factual",
 "verified_by": "human"}
```

`relevant_chunk_ids` is the retrieval ground truth. Unanswerable items set
`answerable: false`, `answer: null`, and an empty `relevant_chunk_ids`; the
correct behaviour on those is to abstain.

### Categories

`qtype` splits the set so a single average can't hide a weak area:

| qtype | what it tests |
| --- | --- |
| `factual` | a plain question the docs answer directly |
| `exact_identifier` | an exact token like `status_code` or `Depends`, where BM25 should beat embeddings |
| `paraphrase` | wording that avoids the docs' own vocabulary, where embeddings should beat BM25 |
| `multi_hop` | needs two or more chunks to answer |
| `near_miss` | the top chunks look relevant but don't contain the answer |
| `unanswerable_in_domain` | a FastAPI question the docs don't cover |
| `unanswerable_out_of_domain` | not about FastAPI at all |

Every metric is reported overall and per category, under `per_category`.

## Retrieval metrics

All four are deterministic — no LLM, no API key, no cost — and are computed from
what retrieval actually returned.

| metric | definition |
| --- | --- |
| `recall_at_k` | fraction of the item's relevant chunks found in the top *k* |
| `all_relevant_at_k` | 1.0 only when *every* relevant chunk is in the top *k* |
| `context_precision_at_k` | fraction of the top *k* that is labelled relevant |
| `mrr` | 1 / rank of the **first** relevant chunk |

**Multiple correct chunks.** A `multi_hop` item lists every chunk needed to
answer it. `recall_at_k` gives partial credit: finding one of two labelled
chunks scores 0.5. That's deliberate — partial retrieval is partially useful —
but it means a multi-hop item can score 0.5 while being unanswerable in
practice. `all_relevant_at_k` is the strict companion that answers "could it
have answered at all", and it's the one to read for `multi_hop`.

MRR still uses the first relevant hit. Averaging a reciprocal rank over several
correct chunks makes the number harder to interpret without telling you anything
`recall_at_k` and `all_relevant_at_k` don't already say.

**Why `context_precision_at_5` looks bad.** It's bounded above by
`|relevant| / k`. With one labelled chunk and `k=5`, 0.2 is a perfect score.
It measures how much of the returned context was labelled, not answer quality.

**Retrieval is scored even when the generator abstains.** An abstention on an
answerable question used to force recall, precision and MRR to zero for that
item. That conflates a retrieval failure with a generation failure and hides
which stage actually broke — retrieval may have returned the right chunk and the
model refused anyway. Abstention has its own metric instead.

## Robustness metrics

| metric | definition |
| --- | --- |
| `false_answer_rate` | share of **unanswerable** items that got an answer instead of an abstention |
| `false_abstention_rate` | share of **answerable** items that got an abstention |

These are the two sides of the same dial, which is why both are reported: a
system can drive `false_answer_rate` to zero by abstaining on everything, and
`false_abstention_rate` is what catches that.

## Judge metrics

Graded by an LLM, so they're noisy and need more care than the deterministic
ones.

| metric | definition |
| --- | --- |
| `faithfulness` | are the answer's claims supported by the retrieved sources? |
| `answer_relevance` | does the answer address the question? |
| `citation_precision` | does each cited chunk back the sentence citing it? |

`citation_validity` sits next to these but needs no judge: it's the share of the
model's citation *attempts* that name a chunk that was actually retrieved. The
model invents anchors that look like documentation headings
(`tutorial/middleware#adding-asgi-middlewares`) rather than chunk indices
(`tutorial/middleware#3`). Those used to be dropped silently during parsing, so
an answer citing two invented chunks was indistinguishable from one citing
nothing, and `citation_precision` couldn't see them because nothing survived to
grade. Read the two together: precision tells you whether the surviving
citations were apt, validity tells you how many survived at all.

**The judge is a different model from the generator.** A model grading its own
output rates itself generously. By default `llm.resolve_judge()` picks a second
model on whichever provider you're already using, so one API key covers both
roles; `JUDGE_PROVIDER` and `JUDGE_MODEL` override it. If nothing else is
available it falls back to the generator's own model and the report records
`judge.independent: false` — read faithfulness with suspicion when you see that.

Pointing the judge at a free tier does not end well: one run needs roughly 190
judge calls, against limits of 20 requests/day (Gemini) or 200k tokens/day
(Groq). The run won't crash, it will report `judge_coverage` below 1.0 and null
the metric.

**Abstentions are not graded.** An abstention asserts nothing, so "are its
claims supported" has no answer; scoring it 0 or 1 would be inventing one.
`metadata.judge_graded` records how many items each judge metric actually
covered, so the size of that exclusion is visible rather than implied.

**Repeated runs.** `--judge-runs N` grades the same stored answers N times and
reports mean, min, max and stdev under `judge_spread`. Be careful what you
conclude from it: at `temperature=0` over identical answers the judge is
deterministic, and measured spread across three passes was exactly 0.0000 for
all three metrics. That means the metric is reproducible, not that it's stable
under change — the variation a CI margin has to absorb comes from regenerating
answers, which needs repeated *full* runs to measure.

If a repeated run does show spread, check `judge_coverage` first. Dropped calls
change the denominator between passes and produce spread that looks like judge
noise but isn't.

**Replaying answers.** `--rejudge eval_items.jsonl` re-grades answers from a
previous run instead of calling the generator again, so repeated judging costs
only the judge's own calls. Paired with `--no-judge` it recomputes the
deterministic metrics for free.

## Confidence intervals

`confidence_intervals` holds a 95% percentile bootstrap interval (2000
resamples, seeded) for each headline metric. With a set of this size one
question is worth about four points of recall, so a bare point estimate invites
reading noise as progress. Quote the interval alongside the number.

A bootstrap interval describes sampling variation in *this* golden set. It does
not account for the set being unrepresentative of real questions, which is a
separate and larger problem.

## Provenance

`metadata` records what produced the numbers: the generator provider and model,
the judge provider and model, whether the judge was independent, the docs
revision the corpus was built from, and the chunk count. A baseline without this
can't be compared against anything — swapping the model looks exactly like a
quality change.

## Running it

```
python eval/evaluate.py --golden data/eval/golden_set.jsonl --out eval_report.json
python eval/evaluate.py --judge-runs 3        # measure judge noise
python eval/evaluate.py --no-judge            # deterministic metrics only, no API calls
```

Per-item results go to `eval_items.jsonl`, which is what the judge-calibration
sampler reads.
