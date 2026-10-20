# Does hybrid retrieval actually beat either half?

**Report:** `reports/ablation.json` · **Script:** `eval/ablation.py` · 2026-10-21

The README has claimed since the start that fusing dense and BM25 beats either
one alone. Nothing had measured it. This is that measurement, plus a sweep of
how many fused candidates are worth handing to the cross-encoder.

## Setup

- 26 golden questions, 20 of which carry labelled chunk ids
- 953 chunks from the FastAPI docs
- `sentence-transformers/all-MiniLM-L6-v2` embeddings in Qdrant,
  `cross-encoder/ms-marco-MiniLM-L6-v2` reranker, both local
- top 20 from each retriever, RRF constant 60
- **no LLM calls** — every metric here is computed from chunk ids, so the
  experiment is free and repeatable

Four variants: `dense` alone, `bm25` alone, `rrf` (fused, no reranking), and
`rrf_rerank` (fused then reordered by the cross-encoder — what ships).

## Results

| variant | recall@5 | 95% CI | recall@20 | MRR | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 0.750 | [0.55, 0.90] | 0.950 | 0.605 | 10 ms | 15 ms |
| bm25 | 0.875 | [0.73, 1.00] | 0.900 | 0.577 | 1 ms | 1 ms |
| rrf | 0.900 | [0.75, 1.00] | 0.950 | 0.703 | 6 ms | 9 ms |
| rrf_rerank | 0.900 | [0.75, 1.00] | 0.950 | **0.785** | 257 ms | 297 ms |

### What the reranker is actually for

It adds no recall at all — 0.900 either way. What it does is move the right
chunk **up** inside the top 5: MRR 0.703 → 0.785. That is what a reranker is
supposed to do, and it is worth knowing that it buys ordering rather than
coverage, because it means a recall regression will never be fixed by tuning it.

It also costs 97% of retrieval latency: 6 ms becomes 257 ms.

### Per category

| category | n | dense | bm25 |
| --- | --- | --- | --- |
| `exact_id` | 2 | 0.500 | **1.000** |
| `multi_hop` | 1 | **1.000** | 0.500 |
| `factual` | 17 | 0.765 | 0.882 |

This is the shape the README predicted — BM25 wins on exact identifiers, dense
wins on the multi-hop question, fusion gets both — and it is also the table
least worth trusting. `exact_id` has two questions and `multi_hop` has one.
Those rows are anecdotes with decimal points, not evidence. They are included
because the per-category split is the whole point of the comparison, and
leaving them out would hide how thin the current set is.

## Rerank depth

How many fused candidates to send to the cross-encoder. The shipped
configuration sends all of them (~40).

| depth | recall@5 | 95% CI | MRR | p50 | p95 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.950 | [0.85, 1.00] | 0.796 | **82 ms** | 83 ms |
| 20 | 0.950 | [0.85, 1.00] | 0.787 | 153 ms | 154 ms |
| 50 (≈ shipped) | 0.900 | [0.75, 1.00] | 0.785 | 255 ms | 289 ms |

**On quality, this proves nothing.** 0.900 → 0.950 on 20 questions is one
question changing, and the intervals overlap almost entirely. Anyone reporting
"shallower reranking improves recall by 5 points" from this table is reading
noise.

**On latency, it is decisive.** 255 ms → 82 ms is a 3.1× reduction, measured
rather than sampled, and it does not depend on the golden set's size at all.
Reranking 10 candidates is three times faster and no worse on any metric here.

## What I'd change

Set rerank depth to 10, on the latency argument alone. The recall numbers are
not evidence and shouldn't be cited as the reason.

That is a configuration change affecting the shipped system, so it needs a fresh
baseline rather than being folded in quietly.

## Limitations

- **20 labelled questions.** Every recall interval here spans 15–25 points. The
  single largest improvement available to this experiment is more questions.
- **Categories are 1–2 items** outside `factual`, so the per-category split
  cannot currently support a conclusion.
- **No `paraphrase` or `near_miss` questions exist yet.** Those are precisely
  where dense should beat BM25 and where both should struggle, so the comparison
  is missing the cases that would discriminate hardest between the variants.
- **Latency is single-process and warm**, measured on one machine with the
  models already loaded. It describes relative cost between variants, not what a
  user would experience.
- `recall@20` is near-identical across variants (0.90–0.95), which means the
  first-stage retrievers almost always find the chunk *somewhere*. The whole
  contest is about ordering, which is why MRR separates the variants and recall
  does not.
