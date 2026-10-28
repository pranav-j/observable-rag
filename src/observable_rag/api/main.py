"""FastAPI app exposing the RAG pipeline.

/ask costs money per call, so it sits behind per-IP and global rate limits (see
limits.py). The pipeline is built once at startup rather than lazily on the
first request: the models take seconds to load, and paying that on a visitor's
first question is the worst possible moment for it.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..generate.answer import RagPipeline
from ..observability.tracing import init_tracing
from .limits import (
    MESSAGES,
    RateLimiter,
    hash_ip,
    max_question_len,
    too_long_message,
)

# Without this the app's own log lines vanish: uvicorn configures its own
# loggers and leaves everyone else's without a handler, so the rate-limit and
# answer records never reached the container log.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("observable_rag.api")

_pipeline: "RagPipeline | None" = None
_limiter = RateLimiter()
_started = 0.0


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global _pipeline, _started
    init_tracing()  # no-op if Phoenix isn't running
    t0 = time.perf_counter()
    if os.getenv("EAGER_LOAD", "1") != "0":
        # Load the models and index now, so the first visitor doesn't wait for
        # them. Cheap insurance: it turns a slow first request into a slow
        # start, which only the deploy sees.
        _pipeline = RagPipeline()
        try:
            _pipeline._retrieve("warmup", top_n=1)
        except Exception as e:  # noqa: BLE001
            log.warning("warmup retrieval failed: %s", e)
    _started = time.perf_counter()
    log.info("ready in %.1fs", _started - t0)
    yield


app = FastAPI(title="observable-rag", lifespan=lifespan)

STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The demo page. One static file, no build step."""
    return FileResponse(STATIC / "index.html")


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        # with_trace: the demo's whole point is showing what retrieval did.
        _pipeline = RagPipeline(with_trace=True)
    return _pipeline


# The docs this corpus was built from, so a citation can link back to the page
# a reader can actually open. Pinned to the same revision as the index.
DOCS_BASE = "https://fastapi.tiangolo.com/"

EXAMPLES = [
    "How do I declare a dependency with Depends?",
    "What does status_code do on a path operation decorator?",
    "How do I share setup code between endpoints?",
    "How do I return a custom response class for one route?",
    # One the docs genuinely don't cover, so a visitor sees it decline.
    "Does FastAPI include a built-in distributed rate limiter?",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    abstained: bool
    # Everything below is what makes the demo observable without Phoenix.
    invalid_citations: list[str] = []
    contradictory: bool = False
    latency_ms: float | None = None
    trace: dict | None = None
    usage: dict | None = None
    sources: list[dict] = []


def _client_hash(request: Request) -> str:
    """Hashed client address. Honours X-Forwarded-For when behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown")
    return hash_ip(ip)


@app.get("/examples")
def examples() -> dict:
    """Questions worth trying, including one the docs don't answer."""
    return {"examples": EXAMPLES}


@app.get("/health")
def health() -> dict:
    """Liveness for the host's checks. Deliberately does no work."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: are the models and index actually loaded?"""
    ok = _pipeline is not None
    return JSONResponse(
        {"status": "ready" if ok else "loading",
         "requests_left_today": _limiter.remaining_today()},
        status_code=200 if ok else 503)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    who = _client_hash(request)

    limit = max_question_len()
    if len(req.question) > limit:
        return JSONResponse({"detail": too_long_message(limit)}, status_code=413)

    decision = _limiter.check(who)
    if not decision.allowed:
        # Logged by hash, and without the question: rate limiting needs to tell
        # visitors apart, not to keep a record of who asked what.
        log.info("rate limited ip=%s reason=%s", who, decision.reason)
        return JSONResponse(
            {"detail": MESSAGES[decision.reason]},
            status_code=429,
            headers={"Retry-After": str(decision.retry_after)})

    _limiter.record(who)
    out = get_pipeline().answer(req.question)
    log.info("answered ip=%s chars=%d abstained=%s citations=%d cost=%s",
             who, len(req.question), out.abstained, len(out.citations),
             (out.usage or {}).get("cost_usd"))

    trace = out.trace.to_dict() if out.trace is not None else None
    return AskResponse(
        answer=out.answer,
        citations=out.citations,
        abstained=out.abstained,
        invalid_citations=out.invalid_citations,
        contradictory=out.contradictory,
        latency_ms=round(out.latency_ms, 1) if out.latency_ms else None,
        trace=trace,
        usage=out.usage,
        sources=_sources(out, trace),
    )


def _sources(out, trace: "dict | None") -> list[dict]:
    """The chunks shown to the model, each with a link to the page it came from
    and whether the answer actually cited it."""
    rows = []
    for c in (trace or {}).get("chunks", []):
        if c.get("final_rank") is None:
            continue
        rows.append({
            "chunk_id": c["chunk_id"],
            "url": c["url"] or DOCS_BASE,
            "cited": c["chunk_id"] in out.citations,
            "found_by": c["found_by"],
            "rerank_score": c["rerank_score"],
        })
    return rows
