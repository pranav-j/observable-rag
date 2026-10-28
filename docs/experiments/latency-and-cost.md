# Where the time goes, and what a query costs

**Report:** `reports/latency.json` · **Script:** `eval/latency.py` · 2026-10-28

Both numbers used to be unmeasurable. Retrieval reported one `hybrid_search`
span covering embed, dense search, BM25 and fusion together, so no stage could
be attributed. And cost was tiktoken arithmetic multiplied by a price typed from
memory — an estimate wearing a decimal point.

## Setup

104 golden questions, 453-chunk corpus pinned to FastAPI `0.159.1`, warm process
on an M-series Mac. Cold start is measured separately in a fresh subprocess,
because folding it into an average makes every stage look slow.

Retrieval-only mode makes no LLM calls and is free to re-run; `--with-llm` runs
the full pipeline and records the provider's own token counts.

## Retrieval, warm

| stage | p50 | p95 | max |
| --- | --- | --- | --- |
| `dense_search` | 5.9 ms | 13.2 ms | 15.2 ms |
| `bm25_search` | 0.8 ms | 1.3 ms | 1.5 ms |
| `fusion` | 0.0 ms | 0.0 ms | 0.0 ms |
| **`rerank`** | **251.0 ms** | **280.4 ms** | 288.9 ms |
| retrieval total | 259.3 ms | 288.6 ms | — |

**The cross-encoder is 97% of retrieval.** Everything else — embedding the
query, searching 453 vectors, scanning the BM25 index, fusing two rankings —
costs under 7 ms combined. BM25 is nearly free at 0.8 ms, and fusion doesn't
register at all.

That matters because the ablation showed reranking adds **no recall**: 0.900
either way. What it buys is ordering, MRR 0.703 → 0.785. So 97% of retrieval
latency is spent moving the right chunk up inside a top-5 that already contained
it. The rerank-depth sweep is the lever here — reranking 10 fused candidates
instead of all of them was 3.1x faster at no measured cost in quality.

## Cold start

| | |
| --- | --- |
| import | 0.5 s |
| first query (model loading) | 19.5 s |
| **total** | **20.0 s** |

Essentially all of it is loading the embedder and the cross-encoder. This is why
the API loads both at startup rather than lazily: the cost is unavoidable, and
paying it during deployment is better than paying it on a visitor's first
question. It is also why a scale-to-zero host was rejected — see
[the deployment ADR](../adr/0006-deploy-to-a-single-ec2-instance.md).

## Generation and cost

104 questions through the full pipeline, `gpt-5-nano` generating:

| | p50 | p95 | max |
| --- | --- | --- | --- |
| `generate` | 13.2 s | 30.5 s | 51.8 s |
| end to end | 13.6 s | 31.0 s | 52.1 s |
| tokens per query | 4,810 | 8,009 | 11,056 |
| cost per query | $0.000936 | $0.002190 | $0.003354 |

Mean cost $0.001073; the whole 104-question run cost **$0.1116**.

**Generation is 98% of end-to-end time** — 13.2 s against 259 ms of retrieval,
a factor of fifty. Any latency work that isn't about the model call is
rearranging deck chairs, which is the real reason the CI gate no longer tries
to hold end-to-end p95 to a tight margin.

The spread is wide: p95 is 2.3x p50 and the worst query took 52 seconds. That
is the provider, not the pipeline, and it is why four runs of identical code
produced p95 figures between 30 s and 43 s.

`gpt-5-nano` is a reasoning model, so completion tokens include reasoning the
reader never sees. One sampled query spent 2,286 prompt tokens against 2,462
completion tokens — more than half the bill on text nobody reads, and the
reason output pricing dominates here despite a 2,600-token retrieved context.

Prices live in `data/pricing.json` with the date they were checked. A model
missing from that file produces a **null** cost rather than a zero, so an
unpriced call cannot quietly understate a total. The report also stores every
per-query cost, so a percentile can be recomputed without paying for the run
again — which mattered, because the first version of this report printed a p95
of $0.000000 after reusing a percentile helper that rounds to two decimals.

## Limitations

- **One machine, warm, single process.** These are relative costs between
  stages, not what a user on a `t4g.small` experiences. The deployed instance
  has two slower cores.
- **Retrieval timings exclude the store lookup** that turns chunk ids back into
  text; it is a dict access on an already-parsed corpus.
- **`fusion` reading 0.0 ms** is real but uninteresting — it is a dict merge
  over at most 40 ids.
- **Cost is per-provider-call.** It does not include the eval judge, which runs
  separately and roughly triples the cost of a full evaluation run.
- **Cold start varies with disk cache.** 20 s is a warm-disk figure; a container
  starting on a cold page cache will be slower.
