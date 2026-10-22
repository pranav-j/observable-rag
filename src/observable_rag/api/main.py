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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    abstained: bool


def _client_hash(request: Request) -> str:
    """Hashed client address. Honours X-Forwarded-For when behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (
        request.client.host if request.client else "unknown")
    return hash_ip(ip)


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
    log.info("answered ip=%s chars=%d abstained=%s citations=%d",
             who, len(req.question), out.abstained, len(out.citations))
    return AskResponse(answer=out.answer, citations=out.citations,
                       abstained=out.abstained)
