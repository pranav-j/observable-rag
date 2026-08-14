"""Phase 3: the citation-enforcing generation prompt."""

from __future__ import annotations

from ..ingest.chunk import Chunk

ABSTAIN_MESSAGE = "The documentation does not cover this."

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided sources. "
    "Cite every claim inline as [source#chunk], using the tag shown on each source. "
    f"If the sources do not contain the answer, reply exactly: '{ABSTAIN_MESSAGE}' "
    "Do not use outside knowledge."
)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Sources:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]