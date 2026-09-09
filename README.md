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

There's a golden set of about 26 hand-checked questions in
`data/eval/golden_set.jsonl`. It's a mix: straightforward factual questions, a
couple that need two chunks to answer, a few exact-identifier ones, and some that
the docs genuinely don't cover (those are there to check that it abstains instead of
making something up).

```
python eval/evaluate.py --golden data/eval/golden_set.jsonl --out eval_report.json
```

Current numbers:

| metric | value | what it means |
| --- | --- | --- |
| recall@5 | 0.90 | the right chunk is in the top 5 for 90% of questions |
| MRR | 0.78 | and it's usually near the top, not scraping in at rank 5 |
| faithfulness | 0.99 | answers stay grounded in the retrieved chunks |
| answer relevance | 0.98 | answers actually address the question |
| false answer rate | 0.00 | it abstained on every unanswerable question |

Context precision comes out around 0.19, which looks bad but isn't: I label one
correct chunk per question and retrieve 5, so ~1/5 is the ceiling. It measures how
much of what came back was the labeled chunk, not the quality of the answer.

Faithfulness and answer relevance are graded by another LLM, so they move around a
little between runs. That's the reason the CI check (below) looks at changes instead
of fixed pass/fail thresholds.

## The CI check

`.github/workflows/eval-gate.yml` runs the eval on every pull request and fails it
if a metric drops more than a small margin below the committed baseline
(`data/eval/baseline.json`). It's a relative check on purpose. A fixed rule like
"faithfulness must be above 0.95" doesn't work well when the metric is noisy: it
either blocks fine changes or gets quietly tuned around. Comparing to a baseline
catches a change that genuinely made things worse while ignoring normal wobble. The
margins live in `eval/ci_gate.py`.

## A few choices, and why

- Stable chunk IDs. Every chunk has an ID like `tutorial/response-status-code#3`,
  based on its position in a pinned copy of the docs. The golden set points at those
  IDs, so if the IDs ever shift, the eval silently breaks. That's the whole point of
  pinning them: the IDs are the contract between retrieval and evaluation. I learned
  this the annoying way when an early version used the wrong IDs and every retrieval
  metric came back 0.

- Hybrid retrieval. Embeddings understand meaning but blur exact tokens; BM25
  matches exact tokens but misses paraphrase. FastAPI docs have a lot of exact
  tokens, so using both and fusing by rank does better than either alone.

- Provider-agnostic LLM. Model providers deprecate models constantly (I got bitten
  by this partway through), so the model name is an environment variable, not
  something hardcoded, and switching providers is just config.

## Layout

```
src/observable_rag/
  ingest/         fetch + chunk the docs (stable IDs)
  index/          vector index (Qdrant) + BM25 index
  retrieve/       RRF fusion + cross-encoder rerank
  generate/       LLM client + citation-enforced answering
  observability/  Phoenix tracing
  api/            FastAPI /ask endpoint
eval/             the eval harness + the CI gate
tests/            run with pytest
scripts/          make_golden.py, drafts golden-set candidates
```

## Notes

- Observability uses Arize Phoenix. Run `python -m phoenix.server.main serve`, then
  run the app or the eval with tracing on, and you get a trace per query
  (retrieve -> rerank -> generate) at http://localhost:6006.
- The golden set is small. A bigger one would give steadier numbers.
  `scripts/make_golden.py` drafts candidates with an LLM, but they still need to be
  checked by hand.
- This is a portfolio project, not a product: one corpus, no auth, no real
  deployment.