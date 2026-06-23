"""Phase 1: fetch and parse the corpus (FastAPI docs) into Documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Document:
    source: str  # stable slug, e.g. "response-status-code"
    url: str
    text: str


def fetch_corpus() -> list[Document]:
    """Download the docs and return parsed Documents. TODO: implement."""
    raise NotImplementedError("phase 1: fetch + parse FastAPI docs")
