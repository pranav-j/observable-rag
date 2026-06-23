"""Phase 4: Phoenix tracing + retrieval-quality logging.

Spans per stage (retrieve / rerank / generate) plus per-query retrieval-quality
records: retrieved chunk ids and scores, whether the cited chunk was retrieved,
and queries that returned nothing useful.
"""

from __future__ import annotations


def init_tracing() -> None:
    """Start the Phoenix tracer and instrument the app. TODO: implement."""
    raise NotImplementedError("phase 4: initialise Arize Phoenix tracing")
