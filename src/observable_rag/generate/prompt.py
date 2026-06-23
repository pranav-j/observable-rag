"""Phase 3: the citation-enforcing generation prompt."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided sources. Cite every claim inline "
    "as [source#chunk]. If the sources do not contain the answer, reply exactly: "
    "'The documentation does not cover this.' Do not use outside knowledge."
)


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    blocks = "\n\n".join(contexts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{blocks}\n\nQuestion: {question}"},
    ]
