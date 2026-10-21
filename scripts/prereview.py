"""Machine pre-review of drafted candidates, to make human review fast.

This does NOT verify anything. It reads each candidate against the chunk it was
drafted from and attaches a verdict -- keep, weak, or drop -- with a reason, so
the human reviewer can spend their attention on the ones that need judgement
instead of reading eighty questions from scratch.

    python scripts/prereview.py
    python scripts/prereview.py --only near_miss

The verdict lands in a `prereview` field on each candidate and is shown by
scripts/review_candidates.py. Nothing here writes to the golden set and nothing
here sets verified_by; accepting a candidate is still a human action.

The checker deliberately runs on a different model from the one that drafted the
candidates (JUDGE_PROVIDER/JUDGE_MODEL, same resolution the eval judge uses). A
model marking its own drafts as good is the same failure as a model grading its
own answers, and it is worth about as much.

Per category it asks a different question:

    factual / exact_identifier / paraphrase
        does the cited chunk actually answer this, and is the reference answer right?
    near_miss
        the chunk must NOT answer it -- a near_miss the chunk answers is just a
        mislabelled factual
    multi_hop
        both chunks must be needed -- one that either chunk answers alone is a
        mislabelled factual
    unanswerable_*
        retrieval runs over the whole corpus first, and the top chunks are shown
        to the checker: if the docs do cover it, the label is wrong
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observable_rag.generate.llm import load_judge_llm, resolve_judge  # noqa: E402
from observable_rag.retrieve.store import load_chunks  # noqa: E402

CANDIDATES = Path("data/eval/candidates.jsonl")
VERDICTS = ("keep", "weak", "drop")

BOLD, DIM, GREEN, YELLOW, RED, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")

_SHAPE = ('Reply with ONLY a JSON object: '
          '{"verdict": "keep" | "weak" | "drop", "reason": "<one short sentence>"}')

_GROUNDED = (
    "You are checking a drafted evaluation question for a documentation QA "
    "system. Given the SOURCE the question was written from, decide:\n"
    "- does the SOURCE actually contain the answer?\n"
    "- is the REFERENCE ANSWER accurate according to the SOURCE?\n"
    "- does the QUESTION stand on its own, without referring to the source?\n"
    "- is it a question a real user might plausibly ask?\n"
    "keep = all four hold. weak = usable but vague, trivial, awkwardly worded, or "
    "only partially answered by the source -- use this whenever you are unsure "
    "rather than guessing between keep and drop. drop = the source plainly does "
    "not answer it, the reference answer is wrong, or the question is incoherent. "
    "Read the WHOLE source before deciding it is absent. " + _SHAPE)

_NEAR_MISS = (
    "You are checking a drafted NEAR-MISS evaluation question. A near-miss looks "
    "topically close to the SOURCE but the SOURCE must NOT contain its answer -- "
    "that is the whole point of the category.\n"
    "keep = closely related to the source's topic AND genuinely not answered by it. "
    "weak = unrelated enough that no retrieval system would be tempted, or only "
    "partially unanswered. drop = the SOURCE does answer it (then it is a "
    "mislabelled factual question). " + _SHAPE)

_MULTI_HOP = (
    "You are checking a drafted MULTI-HOP evaluation question. It must require "
    "BOTH sources to answer.\n"
    "keep = genuinely needs both. weak = technically needs both but is contrived. "
    "drop = either source alone answers it (then it is a mislabelled factual "
    "question), or neither does. " + _SHAPE)

_UNANSWERABLE = (
    "You are checking a drafted UNANSWERABLE evaluation question. The claim being "
    "made is that the FastAPI documentation does NOT answer it. Below are the "
    "chunks a retrieval system actually returns for this question.\n"
    "keep = the retrieved sources genuinely do not answer it. weak = they "
    "partially address it, or the question is vague enough to be arguable. "
    "drop = the sources plainly DO answer it, so the label is wrong. " + _SHAPE)


def _parse(text: str) -> dict:
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and obj.get("verdict") in VERDICTS:
                return {"verdict": obj["verdict"],
                        "reason": str(obj.get("reason", ""))[:200]}
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    return {"verdict": "weak", "reason": "checker returned nothing parseable"}


def _user_block(row: dict, chunks: dict, retrieved: "list | None") -> str:
    q = f"QUESTION:\n{row['question']}\n"
    if row["qtype"].startswith("unanswerable"):
        body = "\n\n".join(f"[{c.id}] {c.text[:1200]}" for c in (retrieved or []))
        return f"{q}\nRETRIEVED SOURCES:\n{body}"

    ids = row.get("relevant_chunk_ids") or []
    if not ids and row.get("drafted_from"):
        ids = [s.strip() for s in row["drafted_from"].split("+")]
    # Never truncate the cited source. Cutting it at 1500 characters made the
    # checker report "the source does not answer this" for 15 candidates whose
    # answer sat past the cut -- a confident verdict about text it hadn't read.
    sources = "\n\n".join(f"[{cid}] {chunks[cid].text}"
                          for cid in ids if cid in chunks)
    answer = row.get("answer") or row.get("draft_note") or "(none given)"
    return f"{q}\nREFERENCE ANSWER:\n{answer}\n\nSOURCE(S):\n{sources}"


def _system_for(qtype: str) -> str:
    if qtype == "near_miss":
        return _NEAR_MISS
    if qtype == "multi_hop":
        return _MULTI_HOP
    if qtype.startswith("unanswerable"):
        return _UNANSWERABLE
    return _GROUNDED


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", type=Path, default=CANDIDATES)
    ap.add_argument("--only", help="check a single qtype")
    ap.add_argument("--top-k", type=int, default=5,
                    help="chunks to retrieve for unanswerable candidates")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [r for r in rows if not args.only or r["qtype"] == args.only]
    if not todo:
        print("nothing to check")
        return 1

    provider, model, independent = resolve_judge()
    print(f"checking {len(todo)} candidates with {provider}/{model}")
    if not independent:
        print("  ! the checker is the same model that drafted these; "
              "its approval means little")

    complete = load_judge_llm()
    chunks = load_chunks()
    needs_retrieval = any(r["qtype"].startswith("unanswerable") for r in todo)
    retrieve = None
    if needs_retrieval:
        from observable_rag.retrieve.hybrid import retrieve as _r
        retrieve = _r

    counts: Counter = Counter()
    for i, row in enumerate(todo, start=1):
        retrieved = None
        if row["qtype"].startswith("unanswerable") and retrieve:
            retrieved = retrieve(row["question"], top_n=args.top_k)
        try:
            out = complete([
                {"role": "system", "content": _system_for(row["qtype"])},
                {"role": "user", "content": _user_block(row, chunks, retrieved)}])
            verdict = _parse(out)
        except Exception as e:  # noqa: BLE001
            verdict = {"verdict": "weak", "reason": f"checker failed: {str(e)[:120]}"}

        verdict["checked_by"] = f"{provider}/{model}"
        if retrieved is not None:
            verdict["retrieved"] = [c.id for c in retrieved]
        row["prereview"] = verdict
        counts[verdict["verdict"]] += 1
        colour = {"keep": GREEN, "weak": YELLOW, "drop": RED}[verdict["verdict"]]
        print(f"  [{i}/{len(todo)}] {row['id']} {row['qtype']:26s} "
              f"{colour}{verdict['verdict']:5s}{RESET} {DIM}{verdict['reason'][:70]}{RESET}")

    with args.candidates.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n{BOLD}{counts['keep']} keep, {counts['weak']} weak, "
          f"{counts['drop']} drop{RESET} -> {args.candidates}")
    print("These are suggestions from a model, not verification. Accepting a "
          "candidate is still yours:\n  python scripts/review_candidates.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
