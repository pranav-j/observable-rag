"""Phase 4: Phoenix tracing + retrieval-quality logging.

Spans wrap each pipeline stage (retrieve / hybrid_search / rerank / generate) so
Phoenix shows the per-query time breakdown and what each stage produced. On top of
generic timing we log RAG-specific retrieval quality -- the retrieved chunk ids and
whether retrieval came back empty -- because RAG failures are almost always
retrieval failures, invisible in latency/token metrics alone.

Tracing is optional. Call init_tracing() to register a Phoenix/OpenTelemetry
tracer; without it, span() is a cheap no-op, so the pipeline runs unchanged in
tests and when Phoenix isn't running. If Phoenix isn't even installed, init_tracing
degrades to a no-op instead of raising.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Optional, Sequence

_TRACER = None  # set by init_tracing()


def tracing_endpoint() -> Optional[str]:
    """Where traces should go, or None if nobody asked for them.

    Tracing is OFF unless an endpoint is configured. It used to default on, and
    registering an exporter succeeds whether or not a collector exists -- so on
    a deployed box with no Phoenix running, every span export retried against
    localhost:4317 and the gRPC retries blocked the request path until the API
    stopped answering. An observability tool must not be able to take down the
    thing it observes.
    """
    return (os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or None)


def init_tracing(project_name: str = "observable-rag",
                 endpoint: Optional[str] = None):
    """Register a Phoenix/OpenTelemetry tracer, if one was asked for.

    Returns None (and stays a no-op) when no endpoint is configured.
    """
    global _TRACER
    endpoint = endpoint or tracing_endpoint()
    if not endpoint:
        print("tracing off (set PHOENIX_COLLECTOR_ENDPOINT to enable)")
        _TRACER = None
        return None
    try:
        from phoenix.otel import register
        provider = register(project_name=project_name, endpoint=endpoint)
        _TRACER = provider.get_tracer(__name__)
        print(f"tracing enabled (project={project_name} -> {endpoint})")
    except Exception as e:  # noqa: BLE001
        print(f"! tracing disabled ({e})")
        _TRACER = None
    return _TRACER


class _NoopSpan:
    def set_attribute(self, *_a, **_k):
        pass


@contextlib.contextmanager
def span(name: str, **attributes) -> Iterator[object]:
    """Open a span if tracing is active; otherwise yield a no-op span."""
    if _TRACER is None:
        yield _NoopSpan()
        return
    with _TRACER.start_as_current_span(name) as sp:
        for k, v in attributes.items():
            try:
                sp.set_attribute(k, v)
            except Exception:  # noqa: BLE001
                pass
        yield sp


def record_retrieval(sp, query: str, retrieved_ids: Sequence[str],
                     scores: Optional[Sequence[float]] = None,
                     gold_ids: Optional[Sequence[str]] = None) -> dict:
    """Attach RAG-specific retrieval quality to the current span; return the attrs."""
    attrs: dict = {
        "retrieval.query": query,
        "retrieval.k": len(retrieved_ids),
        "retrieval.chunk_ids": ", ".join(retrieved_ids),
        "retrieval.empty": len(retrieved_ids) == 0,
    }
    if scores is not None:
        attrs["retrieval.scores"] = ", ".join(f"{s:.3f}" for s in scores)
    if gold_ids is not None:
        attrs["retrieval.gold_hit"] = any(g in retrieved_ids for g in gold_ids)
    for k, v in attrs.items():
        sp.set_attribute(k, v)
    return attrs