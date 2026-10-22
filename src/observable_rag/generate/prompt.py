"""Phase 3: the citation-enforcing generation prompt."""

from __future__ import annotations

from ..ingest.chunk import Chunk

ABSTAIN_MESSAGE = "The documentation does not cover this."

# The earlier version said 'cite every claim inline as [source#chunk]', which
# reads as a template to fill in rather than a tag to copy. The docs themselves
# carry Markdown heading anchors -- `# Middleware { #adding-asgi-middlewares }`
# -- sitting right there in the chunk text, so the model built citations like
# [tutorial/middleware#adding-asgi-middlewares]. Those don't resolve, parsing
# dropped them, and 18 of 41 citations were fabricated. Hence: copy the tag, the
# suffix is a number, and the anchors in the text are not tags.
SYSTEM_PROMPT = (
    "Answer the question using ONLY the provided sources. "
    "Do not use outside knowledge.\n\n"
    "RULE 1 -- Cite every claim inline. Each source below starts with a tag in "
    "square brackets, for example [tutorial/query-params#3]; the part after '#' "
    "is always a NUMBER. Copy a tag EXACTLY as it appears, character for "
    "character. For example: 'Use Depends to declare a dependency "
    "[tutorial/dependencies/index#2].'\n\n"
    "RULE 2 -- ONLY if the sources genuinely do not answer the question, BEGIN "
    f"your reply with exactly this sentence, word for word:\n{ABSTAIN_MESSAGE}\n"
    "That exact opening is how the system records that you declined, so do not "
    "paraphrase it. Use it only when the sources cannot answer the question at "
    "all. If they answer it even partly, ANSWER IT -- never open by declining "
    "and then go on to answer, which contradicts itself. After declining you may "
    "add one sentence pointing at related material the sources do contain.\n\n"
    "Never invent or alter a tag. The source text contains Markdown heading "
    "anchors such as { #adding-asgi-middlewares } -- these are NOT citation tags "
    "and must never appear inside square brackets in your answer. Only ever cite "
    "a tag you can see at the start of one of the sources below."
)


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)


def build_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Sources:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]