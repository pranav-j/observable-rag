"""Draft golden-set candidates with an LLM, for human curation.

Samples chunks from data/corpus/chunks.jsonl and, for each, drafts a question the
chunk answers plus a concise reference answer -- recording that chunk's id as the
supporting evidence (the retrieval ground truth, free). Also drafts a batch of
UNANSWERABLE questions so the eval can measure abstention.

Rate limits: free tiers cap requests-per-minute hard (e.g. 5 RPM). So we BATCH
many excerpts into one request and RETRY on 429 using the server's suggested
wait, instead of firing one request per chunk and dropping rate-limited rows.

Output goes to data/eval/golden_candidates.jsonl -- NOT the golden set. LLM drafts
are noisy, so review, fix, delete weak rows, verify the unanswerable ones truly
aren't covered, and promote the good rows into data/eval/golden_set.jsonl.

    python scripts/make_golden.py --n 60 --unanswerable 15

Uses the provider switch (LLM_PROVIDER, default gemini) and GEMINI_MODEL. Run
`python -m observable_rag.index.build` first so chunks.jsonl exists.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observable_rag.generate.answer import _load_default_llm  # noqa: E402
from observable_rag.retrieve.store import load_chunks  # noqa: E402

CANDIDATES_PATH = "data/eval/golden_candidates.jsonl"


def _answerable_messages(items: list[tuple[int, str]]) -> list[dict]:
    body = "\n\n".join(f"[{i}]\n{text}" for i, text in items)
    system = (
        "You are writing evaluation questions for a documentation QA system. For "
        "EACH numbered excerpt below, write one clear, natural, standalone question "
        "that the excerpt fully answers, plus a concise reference answer (1-2 "
        "sentences). Do not refer to 'the excerpt' or 'the text'. Return ONLY a JSON "
        'array: [{"index": <the excerpt number>, "question": "...", "answer": "..."}] '
        "with one object per excerpt."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": body}]


def _unanswerable_messages(m: int) -> list[dict]:
    return [{"role": "user", "content": (
        f"Write {m} DISTINCT questions that sound like plausible FastAPI questions "
        "but are NOT answerable from FastAPI's official documentation (e.g. features "
        'FastAPI does not have). Return ONLY a JSON array: [{"question": "..."}].')}]


def _parse_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in model response")
    return json.loads(text[start:end + 1])


def _retry_delay(msg: str, default: float) -> float:
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"'?([\d.]+)s'?", msg)
    return float(m.group(1)) + 1 if m else default


def _call(complete, messages, retries: int, pause: float) -> str:
    for attempt in range(retries):
        try:
            return complete(messages)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = _retry_delay(msg, default=max(pause, 12.0))
                print(f"  rate-limited; waiting {wait:.0f}s "
                      f"(attempt {attempt + 1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("gave up after repeated rate-limit errors")


def generate_candidates(chunks, complete, n=60, n_unanswerable=15, seed=0,
                        batch=8, pause=13.0, retries=6) -> list[dict]:
    rng = random.Random(seed)
    sample = rng.sample(list(chunks), min(n, len(chunks)))
    rows: list[dict] = []
    i = 0

    for start in range(0, len(sample), batch):
        group = sample[start:start + batch]
        items = [(local, chunks[cid].text) for local, cid in enumerate(group)]
        try:
            arr = _parse_array(_call(complete, _answerable_messages(items), retries, pause))
        except Exception as e:  # noqa: BLE001
            print(f"! skipped batch @ {start}: {e}", file=sys.stderr)
            continue
        for obj in arr:
            local = obj.get("index")
            if not isinstance(local, int) or not 0 <= local < len(group):
                continue
            q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
            if not q or not a:
                continue
            i += 1
            rows.append({"id": f"gen{i:03d}", "question": q, "answer": a,
                         "relevant_chunk_ids": [group[local]],
                         "answerable": True, "qtype": "factual"})
        time.sleep(pause)

    if n_unanswerable > 0:
        try:
            arr = _parse_array(_call(complete, _unanswerable_messages(n_unanswerable),
                                     retries, pause))
            for obj in arr[:n_unanswerable]:
                q = str(obj.get("question", "")).strip()
                if not q:
                    continue
                i += 1
                rows.append({"id": f"gen{i:03d}", "question": q, "answer": None,
                             "relevant_chunk_ids": [], "answerable": False,
                             "qtype": "unanswerable"})
        except Exception as e:  # noqa: BLE001
            print(f"! unanswerable batch failed: {e}", file=sys.stderr)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft golden-set candidates for curation.")
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--out", default=CANDIDATES_PATH)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--unanswerable", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8, help="excerpts per request")
    ap.add_argument("--sleep", type=float, default=13.0, help="seconds between requests")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    rows = generate_candidates(chunks, _load_default_llm(), n=args.n,
                               n_unanswerable=args.unanswerable, seed=args.seed,
                               batch=args.batch, pause=args.sleep)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} candidates -> {out}")
    print("Now CURATE: review each row, confirm each question matches its cited chunk, "
          "fix answers, delete weak ones, verify the unanswerable ones truly aren't "
          "covered, then promote good rows into data/eval/golden_set.jsonl.")


if __name__ == "__main__":
    main()