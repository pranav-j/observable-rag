"""Phase 1: split Documents into Chunks with STABLE ids.

Chunk ids are the contract the golden set is written against, so the scheme must
not change once the golden set exists: f"{document.source}#{index}", index from 0.

Chunking is block-aware: Markdown is split on blank lines into blocks, fenced
code blocks are kept intact, blocks are packed greedily up to `max_tokens`, and
each chunk after the first is prefixed with the trailing block(s) of the previous
chunk (bounded by `overlap_tokens`) so context isn't lost at boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from .load import Document

_ENC = None


def _default_token_counter(text: str) -> int:
    """Count tokens with tiktoken (cl100k_base); fall back to a word count."""
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENC = False
    if _ENC:
        return len(_ENC.encode(text))
    return max(1, len(text.split()))


@dataclass
class Chunk:
    id: str  # f"{source}#{index}" -- STABLE; do not change after the golden set exists
    source: str
    url: str
    text: str


def _split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    cur: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            cur.append(line)
            in_code = not in_code
            if not in_code:
                blocks.append("\n".join(cur))
                cur = []
            continue
        if in_code:
            cur.append(line)
        elif line.strip() == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b for b in blocks if b.strip()]


def _split_oversized(block: str, max_tokens: int, count) -> list[str]:
    pieces: list[str] = []
    cur: list[str] = []
    ct = 0
    for line in block.split("\n"):
        lt = count(line) or 1
        if cur and ct + lt > max_tokens:
            pieces.append("\n".join(cur))
            cur, ct = [], 0
        cur.append(line)
        ct += lt
    if cur:
        pieces.append("\n".join(cur))
    return pieces or [block]


def chunk_document(doc: Document, max_tokens: int = 512, overlap_tokens: int = 64,
                   count_tokens=None) -> list[Chunk]:
    count = count_tokens or _default_token_counter

    expanded: list[str] = []
    for b in _split_blocks(doc.text):
        expanded.extend(_split_oversized(b, max_tokens, count)
                        if count(b) > max_tokens else [b])

    groups: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for b in expanded:
        bt = count(b)
        if cur and cur_tok + bt > max_tokens:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(b)
        cur_tok += bt
    if cur:
        groups.append(cur)

    if overlap_tokens > 0 and len(groups) > 1:
        with_overlap = [groups[0]]
        for i in range(1, len(groups)):
            carry: list[str] = []
            ct = 0
            for b in reversed(groups[i - 1]):
                t = count(b)
                if carry and ct + t > overlap_tokens:
                    break
                carry.insert(0, b)
                ct += t
            with_overlap.append(carry + groups[i])
        groups = with_overlap

    return [Chunk(id=f"{doc.source}#{i}", source=doc.source, url=doc.url,
                  text="\n\n".join(g)) for i, g in enumerate(groups)]
