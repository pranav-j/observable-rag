# observable-rag

Observable RAG — hybrid retrieval, citation-enforced answers, CI-gated evaluation.

A retrieval-augmented question-answering system over a bounded technical-docs
corpus (the FastAPI docs). Built to behave like a production system rather than a
demo: every answer cites its source, every query is traced end to end, and a
golden-set evaluation suite gates pull requests so quality can't silently regress.

## Why this exists
Most RAG projects stop at "load docs, embed, call an LLM." The parts that
separate a production system — evaluation, observability, regression gating — are
the focus here.

## Architecture
Query -> hybrid retrieval (dense vector + BM25 lexical) -> cross-encoder rerank ->
LLM generation with enforced inline citations, or an abstention when the docs
don't cover the question. Phoenix traces every stage; an offline harness scores
the same pipeline and a CI gate blocks regressions.

## Stack
- API: FastAPI
- Vectors: Qdrant (self-hosted) + local sentence-transformers embeddings
- Lexical: BM25 (rank-bm25)
- Rerank: cross-encoder (sentence-transformers), local
- Generation: OpenAI (the one paid dependency)
- Tracing: Arize Phoenix (local)
- Eval: RAGAS + custom retrieval metrics, gated in GitHub Actions

## Quickstart
    pip install -e ".[eval,dev]"
    docker compose up -d qdrant
    python -m observable_rag.index.build          # phase 1+: fetch + index corpus
    uvicorn observable_rag.api.main:app --reload  # the API
    python eval/evaluate.py --out eval_report.json
    python eval/ci_gate.py --current eval_report.json

The harness and gate run today against a stub pipeline (everything is wired; the
retrieval/generation internals are the build-out below).

## Evaluation
`data/eval/golden_set.jsonl` holds question / answer / source triples, including
unanswerable questions where the correct behaviour is to abstain — that is how
hallucination is measured. The CI gate compares each run to
`data/eval/baseline.json` using relative margins (a metric may not drop more than
N points), not an absolute threshold.

Generate the first real baseline once the pipeline works:
    python eval/evaluate.py --out eval_report.json
    cp eval_report.json data/eval/baseline.json

## Status / roadmap
- [x] Scaffold: structure, eval harness, CI gate, RRF fusion, FastAPI app shell
- [ ] Phase 1 — ingestion + chunking (stable chunk ids)
- [ ] Phase 2 — vector + BM25 indexes, hybrid fusion, cross-encoder rerank
- [ ] Phase 3 — citation-enforcing generation + the /ask endpoint
- [ ] Phase 4 — Phoenix tracing + retrieval-quality logging
- [ ] Phase 5 — containerise + deploy; author and verify the full golden set
