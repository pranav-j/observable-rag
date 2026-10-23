"""Phase 4 tracing tests -- no Phoenix needed (span() is a no-op until wired)."""

import builtins

from observable_rag.observability import tracing


def test_span_is_noop_without_init():
    with tracing.span("stage", foo="bar") as sp:
        sp.set_attribute("k", 1)  # must not raise when tracing is off


def test_record_retrieval_flags_gold_hit_and_empty():
    seen = {}

    class FakeSpan:
        def set_attribute(self, k, v):
            seen[k] = v

    hit = tracing.record_retrieval(FakeSpan(), "q", ["a#0", "b#0"], gold_ids=["b#0"])
    assert hit["retrieval.gold_hit"] is True
    assert hit["retrieval.empty"] is False
    assert hit["retrieval.k"] == 2

    miss = tracing.record_retrieval(FakeSpan(), "q", [], gold_ids=["b#0"])
    assert miss["retrieval.empty"] is True
    assert miss["retrieval.gold_hit"] is False


def test_init_tracing_degrades_without_phoenix(monkeypatch):
    real_import = builtins.__import__

    def no_phoenix(name, *a, **k):
        if name.startswith("phoenix"):
            raise ImportError("phoenix not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_phoenix)
    assert tracing.init_tracing() is None
    tracing._TRACER = None

def test_tracing_is_off_unless_an_endpoint_is_configured(monkeypatch):
    """A registered exporter with no collector behind it retried against
    localhost:4317 until the deployed API stopped answering requests. An
    observability tool must not be able to take down what it observes."""
    import observable_rag.observability.tracing as t

    for var in ("PHOENIX_COLLECTOR_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(t, "_TRACER", "pretend-a-tracer-was-registered")

    assert t.tracing_endpoint() is None
    assert t.init_tracing() is None
    assert t._TRACER is None

    with t.span("x") as sp:          # still a safe no-op
        sp.set_attribute("k", "v")


def test_an_explicit_endpoint_is_honoured(monkeypatch):
    import observable_rag.observability.tracing as t
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://collector:4317")
    assert t.tracing_endpoint() == "http://collector:4317"
