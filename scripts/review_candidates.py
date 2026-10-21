"""Terminal review for drafted golden-set candidates.

Candidates come from an LLM and are wrong often enough that none of them belong
in the golden set unreviewed. This shows one candidate at a time next to the
chunk text it was drafted from, and you accept, edit, or reject it. Accepted
items are appended to data/eval/golden_set.jsonl with verified_by: human --
which is the only way anything gets that field.

    python scripts/review_candidates.py
    python scripts/review_candidates.py --only near_miss
    python scripts/review_candidates.py --candidates data/eval/candidates.jsonl

Decisions are recorded in data/eval/review_log.jsonl, so quitting part-way and
coming back later picks up where you left off. Re-running never re-asks about a
candidate you already decided on.

After a session, refresh the integrity manifest so the new items are covered:

    python -m observable_rag.ingest.manifest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observable_rag.retrieve.store import load_chunks  # noqa: E402

CANDIDATES = Path("data/eval/candidates.jsonl")
GOLDEN = Path("data/eval/golden_set.jsonl")
REVIEW_LOG = Path("data/eval/review_log.jsonl")
CHUNKS = "data/corpus/chunks.jsonl"

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")

HELP = f"""
  {BOLD}a{RESET} accept      {BOLD}e{RESET} edit then accept   {BOLD}r{RESET} reject
  {BOLD}s{RESET} skip        {BOLD}f{RESET} show full chunk    {BOLD}?{RESET} help
  {BOLD}q{RESET} quit (progress is saved)
"""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # golden_set.jsonl has no trailing newline; appending blind would glue the
    # new row onto the last one and silently corrupt both.
    needs_newline = path.exists() and path.stat().st_size > 0 and \
        not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(("\n" if needs_newline else "") + json.dumps(row) + "\n")


def question_key(text: str) -> str:
    """Stable identity for a candidate: its question, normalised.

    NOT the gen0NN id. Those are assigned by position and get renumbered every
    time candidates are redrafted, so a log keyed on them starts skipping
    questions you have never seen while re-asking ones you already decided.
    """
    return " ".join((text or "").lower().split())


def decided_questions(log: list[dict], golden: list[dict]) -> set[str]:
    """Everything already ruled on: logged decisions plus anything in the golden
    set, so an accepted item is never offered twice even if the log is lost."""
    seen = {question_key(d["question"]) for d in log if d.get("question")}
    seen |= {question_key(g["question"]) for g in golden if g.get("question")}
    return seen


def next_golden_id(golden: list[dict]) -> str:
    """Continue the q0NN sequence rather than reusing the draft's gen0NN id."""
    used = [int(g["id"][1:]) for g in golden
            if g.get("id", "").startswith("q") and g["id"][1:].isdigit()]
    return f"q{max(used, default=0) + 1:03d}"


def _wrap(text: str, width: int = 88, limit: "int | None" = 1200) -> str:
    shown = text if limit is None or len(text) <= limit else text[:limit] + " [...]"
    out = []
    for para in shown.split("\n"):
        while len(para) > width:
            cut = para.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(para[:cut])
            para = para[cut:].lstrip()
        out.append(para)
    return "\n".join(out)


def _validate(item: dict, chunks: dict) -> list[str]:
    problems = []
    if not item.get("question", "").strip():
        problems.append("question is empty")
    if item.get("answerable"):
        if not item.get("relevant_chunk_ids"):
            problems.append("answerable but cites no chunk")
        if not item.get("answer"):
            problems.append("answerable but has no reference answer")
    else:
        if item.get("relevant_chunk_ids"):
            problems.append("unanswerable but cites chunks")
        if item.get("answer") is not None:
            problems.append("unanswerable but has an answer")
    for cid in item.get("relevant_chunk_ids", []):
        if cid not in chunks:
            problems.append(f"chunk id not in the corpus: {cid}")
    return problems


def _edit(item: dict) -> "dict | None":
    """Open the candidate in $EDITOR. Returns None if the edit left it unparseable."""
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as fh:
        json.dump(item, fh, indent=2)
        fh.flush()
        path = fh.name
    try:
        subprocess.call([editor, path])
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as e:
        print(f"{RED}  not valid JSON after editing ({e}); candidate left undecided{RESET}")
        return None
    finally:
        os.unlink(path)


def _show(item: dict, chunks: dict, index: int, total: int, full: bool) -> None:
    print(f"\n{BOLD}{'─' * 88}{RESET}")
    print(f"{BOLD}[{index}/{total}]{RESET}  {DIM}{item.get('id')}{RESET}  "
          f"{YELLOW}{item.get('qtype')}{RESET}  "
          f"{'answerable' if item.get('answerable') else 'UNANSWERABLE'}")

    # A machine pre-read, to point attention at the ones that need it. It is a
    # suggestion from a model that did not write this candidate -- not a check
    # that has happened. Accepting is still yours.
    pre = item.get("prereview")
    if pre:
        colour = {"keep": GREEN, "weak": YELLOW, "drop": RED}.get(
            pre.get("verdict"), DIM)
        print(f"\n  {colour}machine read: {pre.get('verdict', '?')}{RESET} "
              f"{DIM}-- {pre.get('reason', '')}{RESET}")
        print(f"  {DIM}({pre.get('checked_by', 'unknown checker')}; "
              f"not verification){RESET}")
    print(f"\n{BOLD}Q:{RESET} {_wrap(item.get('question', ''))}")
    if item.get("answer"):
        print(f"\n{BOLD}A:{RESET} {_wrap(item['answer'])}")
    if item.get("draft_note"):
        print(f"\n{DIM}why it shouldn't be answerable: {item['draft_note']}{RESET}")

    cited = item.get("relevant_chunk_ids") or []
    source = item.get("drafted_from")
    to_show = cited or ([s.strip() for s in source.split("+")] if source else [])
    for cid in to_show:
        chunk = chunks.get(cid)
        label = "cited" if cid in cited else "drafted from (not cited)"
        print(f"\n{BOLD}── {cid}{RESET} {DIM}({label}){RESET}")
        print(_wrap(chunk.text, limit=None if full else 1200) if chunk
              else f"{RED}  missing from the corpus{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Review drafted golden-set candidates.")
    ap.add_argument("--candidates", type=Path, default=CANDIDATES)
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--log", type=Path, default=REVIEW_LOG)
    ap.add_argument("--chunks", default=CHUNKS)
    ap.add_argument("--only", help="review a single qtype")
    ap.add_argument("--verdict", choices=("keep", "weak", "drop"),
                    help="only candidates the pre-read gave this verdict")
    ap.add_argument("--order", choices=("file", "best-first", "worst-first"),
                    default="file", help="sort by the pre-read verdict")
    args = ap.parse_args()

    candidates = _read_jsonl(args.candidates)
    if not candidates:
        print(f"no candidates in {args.candidates}; run scripts/make_golden.py first")
        return 1

    log = _read_jsonl(args.log)
    golden = _read_jsonl(args.golden)
    decided = decided_questions(log, golden)
    queue = [c for c in candidates if question_key(c.get("question", "")) not in decided
             and (not args.only or c.get("qtype") == args.only)
             and (not args.verdict
                  or (c.get("prereview") or {}).get("verdict") == args.verdict)]
    if args.order != "file":
        rank = {"keep": 0, "weak": 1, "drop": 2}
        queue.sort(key=lambda c: rank.get(
            (c.get("prereview") or {}).get("verdict"), 1),
            reverse=args.order == "worst-first")
    if not queue:
        print(f"nothing left to review ({len(decided)} already decided)")
        return 0

    chunks = load_chunks(args.chunks)
    print(f"{len(queue)} to review, {len(decided)} already decided. {HELP}")

    accepted = rejected = skipped = 0
    full = False

    for i, cand in enumerate(queue, start=1):
        while True:
            _show(cand, chunks, i, len(queue), full)
            full = False
            try:
                key = input(f"\n{BOLD}[a/e/r/s/f/?/q]{RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\ninterrupted; progress saved")
                key = "q"

            if key == "?":
                print(HELP)
                continue
            if key == "f":
                full = True
                continue
            if key == "q":
                print(f"\n{accepted} accepted, {rejected} rejected, {skipped} skipped")
                return 0
            if key == "s":
                skipped += 1
                break
            if key == "r":
                _append_jsonl(args.log, {"candidate_id": cand["id"],
                                         "question": cand["question"],
                                         "decision": "reject"})
                rejected += 1
                break
            if key in ("a", "e"):
                item = _edit(cand) if key == "e" else dict(cand)
                if item is None:
                    continue
                problems = _validate(item, chunks)
                if problems:
                    print(f"{RED}  can't accept:{RESET}")
                    for p in problems:
                        print(f"    - {p}")
                    print(f"{DIM}  press e to fix it, or r to reject{RESET}")
                    continue

                entry = {
                    "id": next_golden_id(golden),
                    "question": item["question"].strip(),
                    "answer": item.get("answer"),
                    "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
                    "answerable": bool(item.get("answerable")),
                    "qtype": item["qtype"],
                    "verified_by": "human",
                }
                _append_jsonl(args.golden, entry)
                golden.append(entry)
                _append_jsonl(args.log, {"candidate_id": cand["id"],
                                         "question": cand["question"],
                                         "decision": "accept",
                                         "golden_id": entry["id"]})
                accepted += 1
                print(f"{GREEN}  accepted as {entry['id']}{RESET}")
                break
            print(f"{DIM}  unrecognised key{RESET}")

    print(f"\n{accepted} accepted, {rejected} rejected, {skipped} skipped")
    if accepted:
        print("Refresh the integrity manifest: python -m observable_rag.ingest.manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
