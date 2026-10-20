"""The chunk-id contract: every id the golden set cites must exist in the corpus
and still hold the text it was verified against.

The corpus itself isn't committed (data/corpus/ is ignored), so the corpus-backed
checks skip when the index hasn't been built. The manifest and the golden set ARE
committed, so the checks that only need those two always run.
"""

import json
from pathlib import Path

import pytest

from observable_rag.ingest.manifest import (
    build_manifest,
    check_manifest,
    golden_chunk_ids,
    load_chunk_texts,
    text_hash,
)

CHUNKS = Path("data/corpus/chunks.jsonl")
GOLDEN = Path("data/eval/golden_set.jsonl")
MANIFEST = Path("data/eval/corpus_manifest.json")

needs_corpus = pytest.mark.skipif(
    not CHUNKS.exists(),
    reason="corpus not built; run python -m observable_rag.index.build")


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_covers_every_cited_chunk():
    """A golden item added without refreshing the manifest leaves a gap here."""
    assert set(_manifest()["cited_chunks"]) == set(golden_chunk_ids(GOLDEN))


def test_manifest_hashes_look_like_hashes():
    assert all(len(h) == 16 and h.isalnum()
               for h in _manifest()["cited_chunks"].values())


@needs_corpus
def test_every_golden_chunk_id_exists_in_the_corpus():
    corpus = load_chunk_texts(CHUNKS)
    missing = [c for c in golden_chunk_ids(GOLDEN) if c not in corpus]
    assert not missing, f"golden set cites chunk ids that no longer exist: {missing}"


@needs_corpus
def test_cited_chunk_text_has_not_drifted():
    problems = check_manifest(_manifest(), load_chunk_texts(CHUNKS))
    assert not problems, "\n".join(problems)


@needs_corpus
def test_golden_answers_are_consistent_with_answerability():
    """answerable items cite chunks and carry an answer; unanswerable do neither."""
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if it["answerable"]:
            assert it["relevant_chunk_ids"], f"{it['id']}: answerable but cites no chunk"
            assert it.get("answer"), f"{it['id']}: answerable but has no reference answer"
        else:
            assert not it["relevant_chunk_ids"], f"{it['id']}: unanswerable but cites chunks"
            assert it.get("answer") is None, f"{it['id']}: unanswerable but has an answer"


def test_drift_is_detected():
    manifest = {"fastapi_ref": "x", "n_chunks": 1, "n_cited": 1,
                "cited_chunks": {"a#0": text_hash("original")}}
    assert check_manifest(manifest, {"a#0": "original"}) == []
    assert "text changed" in check_manifest(manifest, {"a#0": "edited"})[0]
    assert "missing from the corpus" in check_manifest(manifest, {})[0]


def test_build_manifest_rejects_uncited_ids():
    with pytest.raises(KeyError):
        build_manifest({"a#0": "text"}, ["a#0", "b#9"], "ref")
