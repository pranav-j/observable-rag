"""Where the time goes, and what a query costs.

    python eval/latency.py                  # retrieval stages only, no LLM calls
    python eval/latency.py --with-llm       # adds generation, and real token cost

Retrieval-only is the default and is free: every stage below the generator is
local, so it can be measured as often as you like. `--with-llm` runs the whole
pipeline and costs roughly $0.07 over the full golden set.

Two things are reported separately on purpose:

  cold start   one process, first query. Model loading dominates it -- the
               first dense search took 13s against 9ms warm -- so folding it
               into an average would make every stage look slow.
  warm p50/p95 steady state, which is what a visitor to a running demo sees.

The reranker was 97% of warm retrieval time before any of this was visible,
because embed, dense search, BM25 and fusion all shared one span.
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

from evaluate import _git_sha, _percentile, load_golden  # noqa: E402

from observable_rag.generate.usage import load_pricing  # noqa: E402


def _pct(values: list[float], q: float, digits: int = 6) -> "float | None":
    """Percentile that keeps small magnitudes.

    evaluate._percentile rounds to 2 decimals because it was written for
    milliseconds; a per-query cost of $0.0011 came back as $0.00, which reads as
    free rather than as a rounding artefact.
    """
    if not values:
        return None
    srt = sorted(values)
    idx = min(len(srt) - 1, int(round((q / 100) * (len(srt) - 1))))
    return round(srt[idx], digits)


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {"p50": _percentile(values, 50), "p95": _percentile(values, 95),
            "mean": round(fmean(values), 2), "max": round(max(values), 2),
            "n": len(values)}


def measure_cold_start() -> dict:
    """Time from a bare process to one answered retrieval, in a subprocess so
    nothing is already loaded."""
    import subprocess
    code = (
        "import time,sys; t=time.perf_counter(); sys.path.insert(0,'src');"
        "from observable_rag.retrieve.hybrid import retrieve;"
        "imported=time.perf_counter();"
        "retrieve('warmup', top_n=5);"
        "print(f'{imported-t:.3f} {time.perf_counter()-imported:.3f}')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=Path.cwd())
    try:
        imported, first = (float(x) for x in out.stdout.strip().split()[-2:])
        return {"import_s": round(imported, 3), "first_query_s": round(first, 3),
                "total_s": round(imported + first, 3)}
    except Exception:  # noqa: BLE001
        return {"error": (out.stderr or "could not measure")[-200:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=Path("data/eval/golden_set.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/latency.json"))
    ap.add_argument("--with-llm", action="store_true",
                    help="also time generation and record real token cost (~$0.07)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    args = ap.parse_args()

    items = load_golden(args.golden)
    if args.limit:
        items = items[: args.limit]

    print("measuring cold start in a fresh process...", flush=True)
    cold = measure_cold_start()
    print(f"  {cold}\n")

    from observable_rag.generate.answer import RagPipeline
    from observable_rag.retrieve.hybrid import retrieve

    stages: dict[str, list[float]] = {}
    end_to_end: list[float] = []
    generate: list[float] = []
    costs: list[float] = []
    tokens: list[int] = []

    pipeline = RagPipeline(with_trace=True) if args.with_llm else None

    retrieve("warmup", top_n=5)          # warm the models before measuring
    print(f"running {len(items)} questions "
          f"({'with' if args.with_llm else 'without'} the LLM)...", flush=True)

    for i, it in enumerate(items, start=1):
        if args.with_llm:
            out = pipeline.answer(it.question)
            trace, usage = out.trace, out.usage or {}
            end_to_end.append(out.latency_ms)
            if usage.get("generate_ms"):
                generate.append(usage["generate_ms"])
            if usage.get("cost_usd") is not None:
                costs.append(usage["cost_usd"])
            if usage.get("total_tokens"):
                tokens.append(usage["total_tokens"])
        else:
            t0 = time.perf_counter()
            _, trace = retrieve(it.question, top_n=5, with_trace=True)
            end_to_end.append((time.perf_counter() - t0) * 1000)
        for stage, ms in trace.stages_ms.items():
            stages.setdefault(stage, []).append(ms)
        if i % 25 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "n_questions": len(items),
            "llm_calls": len(items) if args.with_llm else 0,
            "pricing_checked": load_pricing().get("_checked"),
        },
        "cold_start": cold,
        "warm_ms": {stage: _stats(v) for stage, v in stages.items()},
        "retrieval_total_ms": _stats([sum(v) for v in zip(*stages.values())]),
        "end_to_end_ms": _stats(end_to_end),
    }
    if args.with_llm:
        report["warm_ms"]["generate"] = _stats(generate)
        report["cost"] = {
            "per_query_usd": {"mean": round(fmean(costs), 6) if costs else None,
                              "p50": _pct(costs, 50),
                              "p95": _pct(costs, 95),
                              "max": round(max(costs), 6) if costs else None},
            # Kept so a percentile can be recomputed without paying for the run
            # again, and so the distribution is auditable rather than asserted.
            "per_query_usd_samples": [round(c, 8) for c in costs],
            "tokens_per_query": _stats([float(t) for t in tokens]),
            "total_usd": round(sum(costs), 4) if costs else None,
            "priced_calls": len(costs),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    width = max(len(s) for s in report["warm_ms"])
    print(f"\n{'stage'.ljust(width)}  {'p50':>10} {'p95':>10} {'max':>10}")
    print("-" * (width + 34))
    for stage, st in report["warm_ms"].items():
        print(f"{stage.ljust(width)}  {st['p50']:>9.1f}ms {st['p95']:>9.1f}ms "
              f"{st['max']:>9.1f}ms")
    rt = report["retrieval_total_ms"]
    print(f"{'retrieval total'.ljust(width)}  {rt['p50']:>9.1f}ms {rt['p95']:>9.1f}ms")
    e = report["end_to_end_ms"]
    print(f"{'end to end'.ljust(width)}  {e['p50']:>9.1f}ms {e['p95']:>9.1f}ms")
    if args.with_llm and report["cost"]["per_query_usd"]["mean"] is not None:
        c = report["cost"]
        print(f"\ncost per query: mean ${c['per_query_usd']['mean']:.6f}  "
              f"p95 ${c['per_query_usd']['p95']:.6f}  "
              f"({c['tokens_per_query']['p50']:.0f} tokens p50)")
        print(f"whole run: ${c['total_usd']:.4f} over {c['priced_calls']} calls")
    print(f"\ncold start: {cold.get('total_s', '?')}s "
          f"(import {cold.get('import_s', '?')}s + first query "
          f"{cold.get('first_query_s', '?')}s)")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
