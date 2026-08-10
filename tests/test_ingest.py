"""Phase 1 tests. Chunking is deterministic given text + params, so we inject a
simple word-count tokenizer to keep these offline and fast."""

from observable_rag.ingest.chunk import _split_blocks, chunk_document
from observable_rag.ingest.load import Document


def wc(text: str) -> int:
    return max(1, len(text.split()))


SAMPLE = Document(
    source="tutorial/sample",
    url="https://example.test/",
    text="""# Title

Para one has several words here.

Para two also has a few words.

```python
def f():
    return 1
```

Closing paragraph words.""",
)


def test_ids_are_stable_and_contiguous():
    a = chunk_document(SAMPLE, max_tokens=8, overlap_tokens=0, count_tokens=wc)
    b = chunk_document(SAMPLE, max_tokens=8, overlap_tokens=0, count_tokens=wc)
    ids = [c.id for c in a]
    assert ids == [c.id for c in b]                                  # deterministic
    assert ids == [f"tutorial/sample#{i}" for i in range(len(a))]    # 0-indexed, contiguous


def test_chunks_respect_max_tokens():
    for c in chunk_document(SAMPLE, max_tokens=12, overlap_tokens=0, count_tokens=wc):
        assert wc(c.text) <= 12


def test_fenced_code_block_kept_intact():
    code = [b for b in _split_blocks(SAMPLE.text) if b.strip().startswith("```")]
    assert len(code) == 1
    assert "def f():" in code[0] and "return 1" in code[0]


def test_overlap_prepends_previous_block():
    cs = chunk_document(SAMPLE, max_tokens=10, overlap_tokens=8, count_tokens=wc)
    assert len(cs) >= 2
    assert cs[1].text.split("\n\n")[0] in cs[0].text
