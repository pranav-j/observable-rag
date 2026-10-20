"""Tests for the candidate-review tool's pure parts.

The interactive loop isn't tested, but the three things that would quietly
corrupt the golden set are: id allocation, validation, and appending to a file
that doesn't end in a newline.
"""

import json

from review_candidates import _append_jsonl, _validate, next_golden_id

from observable_rag.ingest.chunk import Chunk

CHUNKS = {"a#0": Chunk(id="a#0", source="a", url="u", text="text")}


def test_next_id_continues_the_sequence():
    assert next_golden_id([{"id": "q001"}, {"id": "q017"}]) == "q018"
    assert next_golden_id([]) == "q001"


def test_next_id_ignores_draft_ids():
    """Candidates carry gen0NN ids; those must not drive the golden sequence."""
    assert next_golden_id([{"id": "q003"}, {"id": "gen099"}]) == "q004"


def test_validate_accepts_a_well_formed_answerable_item():
    item = {"question": "q", "answer": "a", "relevant_chunk_ids": ["a#0"],
            "answerable": True}
    assert _validate(item, CHUNKS) == []


def test_validate_rejects_an_unknown_chunk_id():
    item = {"question": "q", "answer": "a", "relevant_chunk_ids": ["ghost#9"],
            "answerable": True}
    assert "ghost#9" in _validate(item, CHUNKS)[0]


def test_validate_rejects_inconsistent_answerability():
    answerable_without_chunks = {"question": "q", "answer": "a",
                                 "relevant_chunk_ids": [], "answerable": True}
    assert any("cites no chunk" in p
               for p in _validate(answerable_without_chunks, CHUNKS))

    unanswerable_with_answer = {"question": "q", "answer": "a",
                                "relevant_chunk_ids": [], "answerable": False}
    assert any("has an answer" in p
               for p in _validate(unanswerable_with_answer, CHUNKS))


def test_append_does_not_glue_onto_a_file_without_a_trailing_newline(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "q001"}', encoding="utf-8")   # no trailing newline
    _append_jsonl(path, {"id": "q002"})
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r["id"] for r in rows] == ["q001", "q002"]


def test_append_creates_the_file_when_missing(tmp_path):
    path = tmp_path / "new.jsonl"
    _append_jsonl(path, {"id": "q001"})
    assert json.loads(path.read_text())["id"] == "q001"
