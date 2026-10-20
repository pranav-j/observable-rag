"""Retrieval ablation: does hybrid retrieval actually beat either half?

The README has always claimed it does. Nothing measured it. This runs the golden
set through four retrieval configurations and reports recall@5, recall@20, MRR
and latency for each, overall and per category.

    python eval/ablation.py                      # all four variants
    python eval/ablation.py --sweep              # rerank-depth sweep as well
    python eval/ablation.py --out reports/ablation.json

NO LLM CALLS. Every metric here is deterministic and computed from chunk ids, so
this costs nothing and can be re-run freely. The reranker runs locally.

Variants:

    dense       Qdrant vector search alone
    bm25        BM25 lexical search alone
    rrf         the two fused by reciprocal rank, no reranking
    rrf_rerank  fused, then reordered by the cross-encoder (what ships)

The point of the comparison is the per-category split, not the headline. Dense
and BM25 are supposed to fail on different questions -- BM25 on paraphrase,
dense on exact identifiers -- and a single average hides exactly that.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import (  # noqa: E402
    GoldenItem,
    _git_sha,
    _percentile,
    bootstrap_ci,
    load_golden,
    recall_at_k,
    reciprocal_rank,
)

from observable_rag.index import lexical, vector  # noqa: E402
from observable_rag.retrieve.hybrid import reciprocal_rank_fusion  # noqa: E402
from observable_rag.retrieve.rerank import CrossEncoderReranker  # noqa: E402
from observable_rag.retrieve.store import load_chunks  # noqa: E402

VARIANTS = ("dense", "bm25", "rrf", "rrf_rerank")
DEFAULT_SWEEP = (10, 20, 50)


def _rank(variant: str, query: str, top_k_each: int, rrf_k: int,
          store, reranker, rerank_depth: int) -> list[str]:
    """Ranked chunk ids for one variant. Returns the full ranking, not a top-k,
    so recall@5 and recall@20 come from the same call."""
    if variant == "dense":
        return [cid for cid, _ in vector.search(query, top_k_each)]
    if variant == "bm25":
        return [cid for cid, _ in lexical.search(query, top_k_each)]

    dense = [cid for cid, _ in vector.search(query, top_k_each)]
    sparse = [cid for cid, _ in lexical.search(query, top_k_each)]
    fused = reciprocal_rank_fusion([dense, sparse], k=rrf_k)
    if variant == "rrf":
        return fused

    # Rerank only the first `rerank_depth` fused candidates, then keep the rest
    # of the fused order behind them so recall@20 stays measurable.
    head = [store[cid] for cid in fused[:rerank_depth] if cid in store]
    reranked = [c.id for c in reranker.rerank(query, head, top_n=len(head))]
    return reranked + [cid for cid in fused[rerank_depth:]]


def run_variant(variant: str, items: list[GoldenItem], store, reranker,
                top_k_each: int, rrf_k: int, rerank_depth: int) -> dict:
    answerable = [it for it in items if it.answerable and it.relevant_chunk_ids]
    rows = []
    latencies = []
    for it in answerable:
        t0 = time.perf_counter()
        ranking = _rank(variant, it.question, top_k_each, rrf_k, store,
                        reranker, rerank_depth)
        latencies.append((time.perf_counter() - t0) * 1000)
        rows.append({
            "qtype": it.qtype,
            "recall_at_5": recall_at_k(it.relevant_chunk_ids, ranking, 5),
            "recall_at_20": recall_at_k(it.relevant_chunk_ids, ranking, 20),
            "rr": reciprocal_rank(it.relevant_chunk_ids, ranking),
        })

    def agg(subset):
        if not subset:
            return None
        return {
            "recall_at_5": round(fmean(r["recall_at_5"] for r in subset), 4),
            "recall_at_20": round(fmean(r["recall_at_20"] for r in subset), 4),
            "mrr": round(fmean(r["rr"] for r in subset), 4),
            "n": len(subset),
        }

    return {
        **agg(rows),
        "confidence_intervals": {
            "recall_at_5": bootstrap_ci([r["recall_at_5"] for r in rows]),
            "mrr": bootstrap_ci([r["rr"] for r in rows]),
        },
        "latency_ms": {"p50": _percentile(latencies, 50),
                       "p95": _percentile(latencies, 95)},
        "per_category": {
            qtype: agg([r for r in rows if r["qtype"] == qtype])
            for qtype in sorted({r["qtype"] for r in rows})
        },
    }


def _table(results: dict, title: str, columns: tuple) -> str:
    width = max(len(name) for name in results)
    head = f"{title.ljust(width)}  " + "  ".join(c.rjust(12) for c in columns)
    lines = [head, "-" * len(head)]
    for name, r in results.items():
        cells = []
        for c in columns:
            latency = c in ("p50", "p95")
            value = r["latency_ms"].get(c) if latency else r.get(c)
            if value is None:
                cells.append("—".rjust(12))
            elif latency:
                cells.append(f"{value:>10.0f}ms")
            else:
                cells.append(f"{value:>12.4f}")
        lines.append(f"{name.ljust(width)}  " + "  ".join(cells))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=Path("data/eval/golden_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/ablation.json"))
    ap.add_argument("--top-k-each", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rerank-depth", type=int, default=0,
                    help="fused candidates sent to the cross-encoder; 0 = all")
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep rerank depth over %s" % (DEFAULT_SWEEP,))
    ap.add_argument("--sweep-depths", type=int, nargs="+", default=list(DEFAULT_SWEEP))
    args = ap.parse_args()

    items = load_golden(args.golden)
    store = load_chunks()
    reranker = CrossEncoderReranker()

    # Embedded Qdrant takes an exclusive lock on its storage folder, so an eval
    # running in another terminal makes this fail deep inside the client with a
    # portalocker traceback. Say what's actually wrong instead.
    try:
        vector.search("warmup", 1)
    except RuntimeError as e:
        if "already accessed by another instance" not in str(e):
            raise
        print("data/index/qdrant is locked by another process (an eval run?).\n"
              "Embedded Qdrant allows one process at a time -- wait for it to\n"
              "finish, or point QDRANT_URL at a server for concurrent access.")
        return 1
    depth = args.rerank_depth or (args.top_k_each * 2)

    print(f"{len(items)} golden items, "
          f"{sum(1 for i in items if i.answerable and i.relevant_chunk_ids)} with "
          f"labelled chunks. No LLM calls.\n")

    results = {}
    for variant in VARIANTS:
        print(f"  running {variant} ...", flush=True)
        results[variant] = run_variant(variant, items, store, reranker,
                                       args.top_k_each, args.rrf_k, depth)

    sweep = {}
    if args.sweep:
        for d in args.sweep_depths:
            print(f"  running rrf_rerank @ depth {d} ...", flush=True)
            sweep[str(d)] = run_variant("rrf_rerank", items, store, reranker,
                                        args.top_k_each, args.rrf_k, d)

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "top_k_each": args.top_k_each,
            "rrf_k": args.rrf_k,
            "rerank_depth": depth,
            "embedding_model": vector.MODEL_NAME,
            "rerank_model": reranker.model_name,
            "n_chunks": len(store),
            "llm_calls": 0,
        },
        "variants": results,
        "rerank_depth_sweep": sweep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + _table(results, "variant",
                        ("recall_at_5", "recall_at_20", "mrr", "p50", "p95")))
    if sweep:
        print("\n" + _table(sweep, "depth", ("recall_at_5", "mrr", "p50", "p95")))

    categories = sorted({c for r in results.values() for c in r["per_category"]})
    print(f"\nrecall@5 by category{'':<4}" +
          "  ".join(v.rjust(11) for v in VARIANTS))
    for cat in categories:
        cells = []
        for v in VARIANTS:
            per = results[v]["per_category"].get(cat)
            cells.append(f"{per['recall_at_5']:>11.4f}" if per else "—".rjust(11))
        print(f"  {cat:<22}" + "  ".join(cells))

    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
