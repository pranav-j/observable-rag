"""FastAPI app exposing the RAG pipeline (Phase 4: tracing on startup)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from ..generate.answer import RagPipeline
from ..observability.tracing import init_tracing


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    init_tracing()  # no-op if Phoenix isn't running
    yield


app = FastAPI(title="observable-rag", lifespan=lifespan)
_pipeline: "RagPipeline | None" = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    abstained: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    out = get_pipeline().answer(req.question)
    return AskResponse(answer=out.answer, citations=out.citations, abstained=out.abstained)