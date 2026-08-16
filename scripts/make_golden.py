"""Draft golden-set candidates with an LLM, for human curation.

Samples chunks from data/corpus/chunks.jsonl and, for each, asks the LLM to write
a question the chunk answers plus a concise reference answer -- recording that
chunk's id as the supporting evidence (the retrieval ground truth, free). It also
drafts a batch of UNANSWERABLE questions (FastAPI-adjacent but not covered by the
docs) so the eval can measure abstention.

Output goes to data/eval/golden_candidates.jsonl -- NOT the golden set. LLM-drafted
Q&A is noisy, so you review, fix, delete weak rows, verify the unanswerable ones
truly aren't covered, and promote the good rows into data/eval/golden_set.jsonl.
Generate -> curate is the point.

    python scripts/make_golden.py --n 60 --unanswerable 15

Uses the same provider switch as generation (LLM_PROVIDER, default gemini), so it
runs on the free tier. Requires GEMINI_API_KEY (or OPENAI_API_KEY). Run
`python -m observable_rag.index.build` first so chunks.jsonl exists.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# make the package importable when run as a plain script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observable_rag.generate.answer import _load_default_llm  # noqa: E402
from observable_rag.retrieve.store import load_chunks  # noqa: E402

CANDIDATES_PATH = "data/eval/golden_candidates.jsonl"

_ANSWERABLE_PROMPT = (
    "You are writing an evaluation question for a documentation QA system. Given "
    "this documentation excerpt, write ONE clear, natural question a user would ask "
    "that is fully answered by the excerpt, plus a concise reference answer (1-2 "
    "sentences). Ask it as a standalone question -- do NOT refer to 'the excerpt' "
    'or "the text". Return ONLY JSON: {"question": "...", "answer": "..."}'
)
_UNANSWERABLE_PROMPT = (
    "Write ONE question that sounds like a plausible FastAPI question but is NOT "
    "answerable from FastAPI's official documentation (e.g. a feature FastAPI does "
    'not have). Return ONLY JSON: {"question": "..."}'
)


def _parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model response")
    return json.loads(text[start:end + 1])


def generate_candidates(chunks, complete, n=60, n_unanswerable=15, seed=0,
                        pause=0.0) -> list[dict]:
    rng = random.Random(seed)
    sample = rng.sample(list(chunks), min(n, len(chunks)))
    rows: list[dict] = []
    i = 0

    for cid in sample:
        msgs = [{"role": "system", "content": _ANSWERABLE_PROMPT},
                {"role": "user", "content": chunks[cid].text}]
        try:
            obj = _parse_json(complete(msgs))
            i += 1
            rows.append({
                "id": f"gen{i:03d}",
                "question": obj["question"].strip(),
                "answer": obj["answer"].strip(),
                "relevant_chunk_ids": [cid],
                "answerable": True,
                "qtype": "factual",
            })
        except Exception as e:  # noqa: BLE001
            print(f"! skipped {cid}: {e}", file=sys.stderr)
        time.sleep(pause)

    for _ in range(n_unanswerable):
        try:
            obj = _parse_json(complete([{"role": "user", "content": _UNANSWERABLE_PROMPT}]))
            i += 1
            rows.append({
                "id": f"gen{i:03d}",
                "question": obj["question"].strip(),
                "answer": None,
                "relevant_chunk_ids": [],
                "answerable": False,
                "qtype": "unanswerable",
            })
        except Exception as e:  # noqa: BLE001
            print(f"! skipped unanswerable: {e}", file=sys.stderr)
        time.sleep(pause)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft golden-set candidates for curation.")
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--out", default=CANDIDATES_PATH)
    ap.add_argument("--n", type=int, default=60, help="answerable questions to draft")
    ap.add_argument("--unanswerable", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.0,
                    help="seconds between LLM calls (stay under free-tier RPM)")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    rows = generate_candidates(chunks, _load_default_llm(), n=args.n,
                               n_unanswerable=args.unanswerable, seed=args.seed,
                               pause=args.sleep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} candidates -> {out}")
    print("Now CURATE: review each row, fix answers/ids, delete weak ones, verify the "
          "unanswerable ones truly aren't covered, then promote good rows into "
          "data/eval/golden_set.jsonl.")


if __name__ == "__main__":
    main()