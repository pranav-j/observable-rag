"""FastAPI app exposing the RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="observable-rag")


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
    # TODO phase 3: call RagPipeline().answer(req.question) and map the result.
    raise NotImplementedError("phase 3: wire RagPipeline into /ask")
