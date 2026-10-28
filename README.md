# observable-rag

A question-answering system over the FastAPI documentation. You ask something, it
finds the relevant parts of the docs, and it answers with citations that point back
to the source.

I built this mostly to practice the production side of RAG, not just the demo part.
Wiring up "embed some docs and call an LLM" is easy. The parts that are usually
skipped are harder and more interesting: actually measuring whether retrieval and
the answers are any good, being able to look inside a single query when something
goes wrong, and a CI check that fails a pull request if a change makes the numbers
worse. That's where most of the effort went.

The corpus is the FastAPI docs for two reasons. One, I know them well enough to tell
a good answer from a bad one, which matters a lot when you're judging output. Two,
they're full of exact identifiers like `Depends` and `status_code`, where keyword
search beats embeddings, so the hybrid retrieval has a real job to do instead of
being there for show.

## Architecture

The request path when you ask a question:

```
    question
      |
      v
    FastAPI  /ask
      |
      v
    retrieve:
      - vector search (Qdrant, sentence-transformers embeddings)
      - BM25 search (rank-bm25)
      - fuse the two rankings with Reciprocal Rank Fusion
      - cross-encoder rerank, keep the top 5
      |
      v
    generate: LLM answers with inline [source#chunk] citations, or abstains
      |
      v
    cited answer
```

Every stage emits a span to Phoenix, so one query is one trace you can open and look
at: where the time went, which chunks came back, whether it abstained.

Two supporting pieces sit around that path:

```
  build (one-off)     FastAPI docs -> chunks with stable IDs -> Qdrant + BM25 indexes
  eval (offline/CI)   golden set -> eval harness -> metrics -> CI gate vs baseline
```

## How it works

A question goes through two stages.

Retrieval first. Two retrievers run over the docs: a dense vector search
(sentence-transformers embeddings stored in Qdrant) and a BM25 keyword search. Their
results get merged with Reciprocal Rank Fusion, and then a cross-encoder reranker
reorders that merged list and keeps the top few chunks. The idea is that the first
stage is cheap and just needs to get the right chunk somewhere in the top 20, and
the reranker is slower but more accurate and only has to run on those candidates to
push the right one into the top 5.

Then generation. The top chunks are given to an LLM with a prompt that makes it cite
every claim inline as `[source#chunk]`. If the docs don't actually cover the
question, it's supposed to say so rather than invent an answer.

## The demo page

`uvicorn observable_rag.api.main:app` serves a single page at `/` — one static
file, no build step. You type a question and get the answer with its citations
linked back to the FastAPI docs page each chunk came from.

The part I actually care about is below the answer:

- **a stage waterfall** — embed, dense search, BM25, fusion, rerank, generate,
  each with its own time
- **every chunk that was sent to the model**, with which retriever found it
  (dense, BM25, or both), its rank in each, and its cross-encoder score
- **tokens and cost for that request**, from the provider's own counts
- **when it declines**, and when it declines and then answers anyway

That last one matters. It would be easy to show only the good answers; the
failure modes are the interesting part, so they are on screen rather than
hidden. Citations the model invented are rendered outlined instead of quietly
dropped — about 4% of citation attempts don't resolve to a real chunk, and
pretending otherwise would make the demo look better than the system is.

The idea is that someone who never opens Phoenix can still see what retrieval
did and why one chunk beat another.

## Running it

With Docker (easiest):

```
# put your key in .env first, e.g.:
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
docker compose up --build
```

That brings up Qdrant, builds the index from the docs, and then serves the API on
http://localhost:8000. The first run is slow because it downloads a couple of models
and fetches the docs; after that it's cached. Once it's up:

```
curl -X POST http://localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"How do I use Depends for a dependency?"}'
```

Locally without Docker:

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,dev]"
python -m observable_rag.index.build          # fetch the docs + build both indexes
uvicorn observable_rag.api.main:app --reload
```

The LLM is the only thing that costs money, and it's not much (a few cents for a
full eval run). It's provider-agnostic: `LLM_PROVIDER` can be `openai`, `gemini`, or
`groq`, since all three speak the OpenAI API format. I mostly used OpenAI.

## Evaluation

There's a golden set of 104 hand-checked questions in `data/eval/golden_set.jsonl`,
tagged into seven categories so one average can't hide a weak spot: factual,
exact-identifier, paraphrase (worded to avoid the docs' own vocabulary), multi-hop
(needs two chunks), near-miss (looks retrievable but isn't answered), and two kinds
of unanswerable — FastAPI questions the docs don't cover, and questions that aren't
about FastAPI at all. A quarter of the set is unanswerable, which is how you measure
whether it makes things up.

An LLM drafts candidates per category; I review every one by hand against the chunk
it cites before it gets in. Details in [docs/evaluation.md](docs/evaluation.md).

```
python eval/evaluate.py --golden data/eval/golden_set.jsonl --out eval_report.json
```

Current numbers, from `reports/eval_104_2026-10-22.json`:

| metric | value | 95% CI | what it means |
| --- | --- | --- | --- |
| recall@5 | 0.83 | 0.75–0.91 | the right chunk is in the top 5 for 83% of questions |
| all-relevant@5 | 0.81 | 0.72–0.90 | *every* chunk needed is in the top 5 |
| MRR | 0.71 | 0.62–0.79 | and it's usually near the top, not scraping in at rank 5 |
| faithfulness | 0.99 | 0.98–1.00 | answers stay grounded in the retrieved chunks |
| answer relevance | 0.98 | 0.95–1.00 | answers actually address the question |
| citation precision | 0.86 | 0.79–0.93 | each citation backs the sentence it's on |
| false answer rate | 0.27 | 0.12–0.46 | it answered 27% of the unanswerable questions |
| false abstention rate | 0.04 | 0.00–0.09 | it refused 4% of the answerable ones |

These are worse than the numbers I had before, and that's the honest version. The
old set was 26 questions, nearly all straightforward lookups, and it reported
recall@5 0.90 and a false answer rate of 0.00. Four times the questions, including
categories chosen to be hard, and the picture changes. The intervals are half as
wide too, which is the other half of the point.

The per-category breakdown is where it gets useful:

| category | n | recall@5 | false answers |
| --- | --- | --- | --- |
| factual | 42 | 0.91 | — |
| paraphrase | 12 | 0.83 | — |
| exact identifier | 13 | 0.77 | — |
| multi-hop | 11 | 0.64 (all-relevant 0.46) | — |
| unanswerable, not about FastAPI | 8 | — | 0.00 |
| unanswerable, about FastAPI | 13 | — | 0.31 |
| near-miss | 5 | — | **0.60** |

Two things I didn't know before building this set. Multi-hop retrieval finds one of
the two chunks it needs far more often than both — recall@5 0.64 against
all-relevant@5 0.46 — so a recall-only report would have called that fine. And
near-miss is the real weakness: asked something adjacent to a chunk it retrieved but
not actually answered by it, the system answers anyway 60% of the time.

The near-miss failures score **1.00 faithfulness**, which took me a moment. They're
faithful — the answer is a correct restatement of the chunk. It just isn't an answer
to the question. Faithfulness measures grounding, not correctness, and a system that
parrots retrieved text scores perfectly on it. That's worth knowing before trusting
it as a headline number.

Context precision comes out around 0.18, which looks bad but isn't: most questions
have one correct chunk and I retrieve 5, so ~1/5 is the ceiling. It measures how
much of what came back was labeled, not the quality of the answer.

Faithfulness, answer relevance and citation precision are graded by another LLM. It
runs on a different model from the one generating answers — when they were the same
model it was grading its own work, and faithfulness came out 0.993 instead of the
0.95 an independent judge gave the same answers. The judge is not calibrated against
human labels yet, so read those three with that in mind.

### Does hybrid retrieval actually help?

It does, and the per-category numbers say why. From `reports/ablation.json`
(no LLM calls — this is all chunk-ID arithmetic, so it's free to re-run):

| retrieval | recall@5 | MRR | p50 |
| --- | --- | --- | --- |
| dense only | 0.75 | 0.61 | 10 ms |
| BM25 only | 0.88 | 0.58 | 1 ms |
| RRF | 0.90 | 0.70 | 6 ms |
| RRF + rerank | 0.90 | 0.79 | 257 ms |

Dense and BM25 fail on different questions — BM25 gets exact identifiers that dense
blurs, dense handles the multi-hop one that BM25 misses — and fusing recovers both.
The reranker adds no recall at all; what it buys is ordering, moving the right chunk
up inside the top 5, for 97% of the retrieval latency. Write-up in
[docs/experiments/retrieval-ablation.md](docs/experiments/retrieval-ablation.md).

## Latency and cost

From `reports/latency.json` (`python eval/latency.py`, which runs
retrieval-only and free by default). Warm process, 104 questions, M-series Mac:

| stage | p50 | p95 |
| --- | --- | --- |
| dense search | 5.9 ms | 13.2 ms |
| BM25 search | 0.8 ms | 1.3 ms |
| fusion | 0.0 ms | 0.0 ms |
| **rerank** | **251.0 ms** | 280.4 ms |
| retrieval total | 259.3 ms | 288.6 ms |

**The cross-encoder is 97% of retrieval time.** Everything else — embedding the
query, searching the vectors, scanning BM25, fusing two rankings — is under 7 ms
combined.

Put that next to the ablation: reranking adds *no recall*, only ordering. So
almost all of retrieval's cost goes on moving the right chunk higher within a
top-5 that already contained it. The depth sweep says reranking 10 candidates
instead of all of them is about 3x faster for no measured loss, which is the
obvious thing to change next.

Cold start is 20 seconds, nearly all of it loading the two models. That is why
the API loads them at startup instead of on the first request, and why a
scale-to-zero host was the wrong shape for this.

Generation dominates end to end by two orders of magnitude over retrieval, and
cost comes from the provider's own token counts rather than an estimate —
prices live in `data/pricing.json` with the date I last checked them, and a
model that isn't listed produces a null cost rather than a zero. Write-up in
[docs/experiments/latency-and-cost.md](docs/experiments/latency-and-cost.md).

## The CI check

Two workflows. `.github/workflows/ci.yml` needs no secrets, so it runs on pull
requests from forks: lint, unit tests, a Docker build, and a check that every chunk
ID the golden set points at still exists. `.github/workflows/eval-gate.yml` runs the
eval itself and fails a PR if a metric drops more than its margin below the committed
baseline; with no API key available it skips and says so rather than failing with a
traceback.

It's a relative check on purpose. A fixed rule like "faithfulness must be above 0.95"
doesn't work well when the metric is noisy: it either blocks fine changes or gets
quietly tuned around. Comparing to a baseline catches a change that genuinely made
things worse while ignoring normal wobble.

Eleven metrics are gated, and each margin is sized in questions rather than picked by
feel — a 0.03 margin on recall@5 absorbs two questions out of 78. The ones I couldn't
measure are labeled as guesses rather than dressed up:
[docs/gate-margins.md](docs/gate-margins.md).

## A few choices, and why

- Stable chunk IDs. Every chunk has an ID like `tutorial/response-status-code#3`,
  based on its position in the page. The golden set points at those IDs, so if the
  IDs ever shift, the eval silently breaks. I learned this the annoying way when an
  early version used the wrong IDs and every retrieval metric came back 0.

  The catch I hadn't thought about: a stable ID pins *where* a chunk sits, not what
  it says. Add a paragraph near the top of a page and everything below re-packs — the
  ID survives while the text under it becomes something nobody checked. I rebuilt
  from `master` six weeks after the first index and 531 of 952 chunks had different
  text under the same ID. So there's a manifest of hashes for every chunk the golden
  set cites, and a test that fails when one changes. `FASTAPI_REF` isn't pinned to a
  tag yet, which it should be — the manifest catches the drift, but only after it
  happens.

- Hybrid retrieval. Embeddings understand meaning but blur exact tokens; BM25
  matches exact tokens but misses paraphrase. FastAPI docs have a lot of exact
  tokens, so using both and fusing by rank does better than either alone.

- Provider-agnostic LLM. Model providers deprecate models constantly (I got bitten
  by this partway through — a hardcoded default 404'd), so the model name is an
  environment variable, not something hardcoded, and switching providers is just
  config.

- A judge that isn't the generator. The eval's faithfulness score used to come from
  the same model that wrote the answer, which is marking your own homework. It now
  picks a second model on whichever provider you're already using, so one API key
  still covers both. Why it works that way:
  [docs/adr/](docs/adr/).

## Layout

```
src/observable_rag/
  ingest/         fetch + chunk the docs (stable IDs) + integrity manifest
  index/          vector index (Qdrant) + BM25 index
  retrieve/       RRF fusion + cross-encoder rerank + per-stage trace
  generate/       LLM client, citation-enforced answering, token accounting
  observability/  Phoenix tracing (off unless an endpoint is configured)
  api/            FastAPI app, rate limits, and the demo page
eval/             the eval harness, the CI gate, the ablation, latency + cost
tests/            run with pytest
scripts/          drafting, review and judge-calibration tools
reports/          every number in this README comes from a file in here
docs/             evaluation method, experiment write-ups, ADRs
```

## Notes

- Observability uses Arize Phoenix. Run `python -m phoenix.server.main serve`, then
  run the app or the eval with tracing on, and you get a trace per query
  (retrieve -> rerank -> generate) at http://localhost:6006.
- The golden set is 104 questions, which is enough to tell categories apart but not
  enough to be confident within one — near-miss is 5 questions, so that 0.60 false
  answer rate is three questions. `scripts/make_golden.py` drafts more candidates,
  `scripts/review_candidates.py` is the review tool, and nothing reaches the golden
  set without being read by hand.

- The judge isn't calibrated against human labels yet. `scripts/calibrate_judge.py`
  samples answers and computes agreement and Cohen's kappa, but I haven't done the
  labeling, so faithfulness and citation precision are an unchecked second opinion.
- The demo config is the eval config. Same provider, same model, same corpus —
  so the numbers above describe the thing you can actually try, not a different
  setup that happened to score better.
- It has been deployed and taken down again rather than left running: a single
  `t4g.small` in Mumbai, image and index baked in, behind per-IP and global
  rate limits. `scripts/deploy.sh plan` shows what it would create without
  creating it. Reasoning in
  [docs/adr/0006-deploy-to-a-single-ec2-instance.md](docs/adr/0006-deploy-to-a-single-ec2-instance.md).
- This is a portfolio project, not a product: one corpus, no auth, one instance
  with no redundancy.