"""Draft golden-set candidates with an LLM, for human curation.

Samples chunks from data/corpus/chunks.jsonl and drafts questions in each of the
categories the eval reports on (see docs/evaluation.md), because a set that is
all straightforward factual lookups makes retrieval look better than it is. Each
category needs a different prompt:

    factual                     a question the chunk answers directly
    exact_identifier            asks about a literal token, where BM25 should win
    paraphrase                  avoids the chunk's own wording, where dense wins
    multi_hop                   needs two chunks from the same page
    near_miss                   looks like the chunk answers it, but it doesn't
    unanswerable_in_domain      a FastAPI question the docs don't cover
    unanswerable_out_of_domain  not about FastAPI at all

Rate limits: BATCH many excerpts into one request and RETRY on 429 using the
server's suggested wait, so a per-minute cap doesn't drop rows. Progress is
logged per batch so you can see it working.

Output goes to data/eval/candidates.jsonl -- NOT the golden set. LLM drafts are
noisy and the near_miss and unanswerable ones are wrong often enough to matter,
so nothing reaches the golden set without going through:

    python scripts/review_candidates.py

    python scripts/make_golden.py --n 80
    python scripts/make_golden.py --only multi_hop --n 10

Uses the provider switch (LLM_PROVIDER). Run `python -m observable_rag.index.build`
first so chunks.jsonl exists.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observable_rag.generate.answer import _load_default_llm  # noqa: E402
from observable_rag.retrieve.store import load_chunks  # noqa: E402

CANDIDATES_PATH = "data/eval/candidates.jsonl"

# Rough target shape for an 80-question set: every category present, and a bit
# over 15% unanswerable split across the two kinds. --n scales these.
MIX = {
    "factual": 0.30,
    "exact_identifier": 0.15,
    "paraphrase": 0.15,
    "multi_hop": 0.125,
    "near_miss": 0.075,
    "unanswerable_in_domain": 0.10,
    "unanswerable_out_of_domain": 0.10,
}
GROUNDED = ("factual", "exact_identifier", "paraphrase", "near_miss")

# Pages that aren't documentation of the framework. release-notes alone is 500 of
# the corpus's 953 chunks, so uniform sampling drew 45% of candidates from the
# changelog and produced questions like "what translations were added in this
# batch?" -- answerable, and useless for judging a docs QA system. Excluded from
# SAMPLING only; they stay in the index and remain retrievable.
NON_DOC_SOURCES = ("release-notes", "fastapi-people", "newsletter", "management")

# Excerpts are numbered so the model can map answers back to them, and that
# numbering is an invitation: the first run produced twelve questions beginning
# "In excerpt 3, what is...". A question that refers to its own source can't be
# asked of a retrieval system, so every grounded prompt repeats this rule.
_STANDALONE = (
    "Each question must stand completely on its own, as if typed into a search "
    "box by someone who has never seen these excerpts. NEVER write 'in excerpt "
    "N', 'the excerpt', 'the text', 'the passage', 'above', or any other "
    "reference to the source material. "
)
# Excerpts are labelled with their real chunk id, and the model echoes that id
# back, because positional indices did not survive the round trip: questions came
# back attached to a neighbouring excerpt's number, so "how do I declare form
# fields" was recorded as answered by deployment/server-workers#5. An id can be
# checked against the batch; an integer cannot.
_JSON_SHAPE = ('Return ONLY a JSON array: '
               '[{"source": "<the exact SOURCE id of the excerpt, copied '
               'character for character>", "question": "...", "answer": "..."}] '
               "with one object per excerpt.")

_PROMPTS = {
    "factual": (
        "You are writing evaluation questions for a documentation QA system. For "
        "EACH numbered excerpt below, write one clear, natural, standalone question "
        "that the excerpt fully answers, plus a concise reference answer (1-2 "
        "sentences). " + _STANDALONE + _JSON_SHAPE),
    "exact_identifier": (
        "For EACH numbered excerpt, write one question about a LITERAL identifier "
        "that appears in it -- a parameter, decorator, class, or function name such "
        "as status_code or Depends. The question must name that identifier exactly, "
        "the way someone would who already knew the name and wanted to know what it "
        "does. Add a concise reference answer. " + _STANDALONE + _JSON_SHAPE),
    "paraphrase": (
        "For EACH numbered excerpt, write one question it answers, using everyday "
        "wording that AVOIDS the excerpt's own technical vocabulary. If the excerpt "
        "says 'dependency injection', ask about 'sharing setup code between "
        "endpoints'. The question must still be answerable from the excerpt. Add a "
        "concise reference answer. " + _STANDALONE + _JSON_SHAPE),
    "near_miss": (
        "For EACH numbered excerpt, write one question that is CLOSELY RELATED to "
        "its topic and uses similar vocabulary, but that the excerpt does NOT "
        "answer -- a neighbouring detail it never states. These test whether a "
        "retrieval system can tell 'looks relevant' from 'contains the answer'. "
        'Set "answer" to a short note on why the excerpt does not answer it. '
        + _STANDALONE + _JSON_SHAPE),
}

_MULTI_HOP_PROMPT = (
    "You are writing evaluation questions for a documentation QA system. For EACH "
    "numbered PAIR of excerpts below, write one question that genuinely needs BOTH "
    "excerpts to answer -- not one that either answers alone. Add a concise "
    "reference answer that draws on both. If no honest two-part question connects "
    "the pair, omit that index entirely rather than inventing one. "
    + _STANDALONE + _JSON_SHAPE)

_UNANSWERABLE_PROMPTS = {
    "unanswerable_in_domain": (
        "Write {m} DISTINCT questions that sound like plausible FastAPI questions "
        "but are NOT answerable from FastAPI's official documentation -- features "
        "FastAPI does not have, or details it never specifies. "
        'Return ONLY a JSON array: [{{"question": "..."}}].'),
    "unanswerable_out_of_domain": (
        "Write {m} DISTINCT technical questions that have NOTHING to do with "
        "FastAPI or Python web frameworks -- other languages, databases, "
        "infrastructure. They should be the kind of thing someone might type into "
        "the wrong search box. "
        'Return ONLY a JSON array: [{{"question": "..."}}].'),
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _excerpt_messages(qtype: str, items: list[tuple[str, str]]) -> list[dict]:
    body = "\n\n".join(f"SOURCE: {cid}\n{text}" for cid, text in items)
    return [{"role": "system", "content": _PROMPTS[qtype]},
            {"role": "user", "content": body}]


def _pair_messages(pairs: list[tuple[str, str, str]]) -> list[dict]:
    body = "\n\n".join(f"SOURCE: {label}\nEXCERPT A:\n{a}\n\nEXCERPT B:\n{b}"
                       for label, a, b in pairs)
    return [{"role": "system", "content": _MULTI_HOP_PROMPT},
            {"role": "user", "content": body}]


def _unanswerable_messages(qtype: str, m: int) -> list[dict]:
    return [{"role": "user", "content": _UNANSWERABLE_PROMPTS[qtype].format(m=m)}]


def _parse_objects(text: str) -> list[dict]:
    """Extract JSON objects one at a time, tolerating prose, code fences, and
    missing commas between objects. Using the decoder (not a regex) means braces
    inside string values -- e.g. FastAPI's {item_id} -- don't break parsing."""
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    idx, n = 0, len(text)
    while idx < n:
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx = brace + 1
    if not objects:
        raise ValueError("no parseable JSON objects in model response")
    return objects


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
                _log(f"    rate-limited; waiting {wait:.0f}s "
                     f"(attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("gave up after repeated rate-limit errors")


def plan_counts(n: int, only: "str | None" = None) -> dict[str, int]:
    """How many candidates to draft per category."""
    if only:
        return {only: n}
    counts = {q: max(1, round(n * share)) for q, share in MIX.items()}
    return counts


def eligible_chunk_ids(chunks, exclude: "tuple[str, ...]" = NON_DOC_SOURCES) -> list[str]:
    """Chunk ids worth drawing an evaluation question from."""
    return sorted(cid for cid in chunks if cid.split("#")[0] not in exclude)


def _multi_hop_pairs(chunks: dict, rng: random.Random, m: int,
                     exclude: "tuple[str, ...]" = NON_DOC_SOURCES) -> list[tuple[str, str]]:
    """Pairs of chunks from the same page -- the only pairs likely to support an
    honest two-part question."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for cid in eligible_chunk_ids(chunks, exclude):
        by_source[cid.split("#")[0]].append(cid)
    eligible = [sorted(ids) for ids in by_source.values() if len(ids) >= 2]
    rng.shuffle(eligible)
    pairs = []
    for ids in eligible:
        if len(pairs) >= m:
            break
        a, b = rng.sample(ids, 2)
        pairs.append((a, b))
    return pairs


def _draft_grounded(qtype, chunk_ids, chunks, complete, counter, batch, pause,
                    retries) -> list[dict]:
    rows: list[dict] = []
    total = (len(chunk_ids) + batch - 1) // batch
    for b, start in enumerate(range(0, len(chunk_ids), batch), start=1):
        group = chunk_ids[start:start + batch]
        items = [(cid, chunks[cid].text) for cid in group]
        _log(f"  {qtype} batch {b}/{total} ({len(group)}) -> requesting...")
        before = len(rows)
        try:
            arr = _parse_objects(
                _call(complete, _excerpt_messages(qtype, items), retries, pause))
        except Exception as e:  # noqa: BLE001
            _log(f"  ! {qtype} batch {b} skipped: {e}")
            continue
        in_batch = set(group)
        for obj in arr:
            source = str(obj.get("source", "")).strip()
            if source not in in_batch:
                _log(f"    ! dropped: source {source!r} is not in this batch")
                continue
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
            if not q or not a:
                continue
            # near_miss items are deliberately NOT answered by their source chunk,
            # so they carry no ground-truth chunk and no reference answer.
            near_miss = qtype == "near_miss"
            rows.append({
                "id": f"gen{next(counter):03d}",
                "question": q,
                "answer": None if near_miss else a,
                "relevant_chunk_ids": [] if near_miss else [source],
                "answerable": not near_miss,
                "qtype": qtype,
                "drafted_from": source,
                "draft_note": a if near_miss else None,
            })
        _log(f"    +{len(rows) - before} (total {len(rows)})")
        time.sleep(pause)
    return rows


def _draft_multi_hop(chunks, complete, counter, rng, m, batch, pause,
                     retries, exclude=NON_DOC_SOURCES) -> list[dict]:
    pairs = _multi_hop_pairs(chunks, rng, m, exclude)
    rows: list[dict] = []
    total = (len(pairs) + batch - 1) // batch
    for b, start in enumerate(range(0, len(pairs), batch), start=1):
        group = pairs[start:start + batch]
        payload = [(f"{a} + {bb}", chunks[a].text, chunks[bb].text)
                   for a, bb in group]
        _log(f"  multi_hop batch {b}/{total} ({len(group)}) -> requesting...")
        try:
            arr = _parse_objects(_call(complete, _pair_messages(payload),
                                       retries, pause))
        except Exception as e:  # noqa: BLE001
            _log(f"  ! multi_hop batch {b} skipped: {e}")
            continue
        by_label = {f"{a} + {bb}": (a, bb) for a, bb in group}
        for obj in arr:
            label = str(obj.get("source", "")).strip()
            if label not in by_label:
                _log(f"    ! dropped: source {label!r} is not in this batch")
                continue
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
            if not q or not a:
                continue
            rows.append({
                "id": f"gen{next(counter):03d}", "question": q, "answer": a,
                "relevant_chunk_ids": list(by_label[label]),
                "answerable": True, "qtype": "multi_hop",
                "drafted_from": label, "draft_note": None,
            })
        time.sleep(pause)
    return rows


def _draft_unanswerable(qtype, complete, counter, m, pause, retries) -> list[dict]:
    _log(f"  drafting {m} {qtype}...")
    rows: list[dict] = []
    try:
        arr = _parse_objects(
            _call(complete, _unanswerable_messages(qtype, m), retries, pause))
    except Exception as e:  # noqa: BLE001
        _log(f"  ! {qtype} batch failed: {e}")
        return rows
    for obj in arr[:m]:
        q = str(obj.get("question", "")).strip()
        if not q:
            continue
        rows.append({
            "id": f"gen{next(counter):03d}", "question": q, "answer": None,
            "relevant_chunk_ids": [], "answerable": False, "qtype": qtype,
            "drafted_from": None, "draft_note": None,
        })
    _log(f"    +{len(rows)}")
    return rows


def generate_candidates(chunks, complete, counts: dict[str, int], seed=0,
                        batch=8, pause=13.0, retries=6,
                        exclude: "tuple[str, ...]" = NON_DOC_SOURCES) -> list[dict]:
    rng = random.Random(seed)
    counter = iter(range(1, 10_000))
    rows: list[dict] = []

    pool = eligible_chunk_ids(chunks, exclude)
    rng.shuffle(pool)
    cursor = 0

    for qtype in [q for q in MIX if q in counts]:
        m = counts[qtype]
        if m <= 0:
            continue
        if qtype in GROUNDED:
            # A batch sometimes comes back with fewer objects than excerpts sent
            # -- one paraphrase batch of 8 returned 1. Top up from fresh chunks
            # rather than quietly handing over a short category.
            produced = 0
            for attempt in range(3):
                want = m - produced
                if want <= 0:
                    break
                group, cursor = pool[cursor:cursor + want], cursor + want
                if not group:
                    _log(f"  ! out of eligible chunks for {qtype}")
                    break
                got = _draft_grounded(qtype, group, chunks, complete, counter,
                                      batch, pause, retries)
                rows += got
                produced += len(got)
                if len(got) < want and attempt < 2:
                    _log(f"    {qtype}: {produced}/{m}, topping up")
            if produced < m:
                _log(f"  ! {qtype} finished short: {produced}/{m}")
        elif qtype == "multi_hop":
            # The prompt tells the model to omit a pair rather than invent a
            # two-hop question for chunks that don't support one, so a shortfall
            # here is it behaving correctly. Offer more pairs instead of pushing.
            produced = 0
            for attempt in range(3):
                want = m - produced
                if want <= 0:
                    break
                got = _draft_multi_hop(chunks, complete, counter, rng,
                                       want * 2 if attempt else want,
                                       batch, pause, retries, exclude)
                rows += got[:want]
                produced += len(got[:want])
                if produced < m and attempt < 2:
                    _log(f"    multi_hop: {produced}/{m}, offering more pairs")
            if produced < m:
                _log(f"  ! multi_hop finished short: {produced}/{m}")
        else:
            rows += _draft_unanswerable(qtype, complete, counter, m, pause, retries)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Draft golden-set candidates for curation.")
    ap.add_argument("--chunks", default="data/corpus/chunks.jsonl")
    ap.add_argument("--out", default=CANDIDATES_PATH)
    ap.add_argument("--n", type=int, default=80,
                    help="total candidates to draft, split across categories")
    ap.add_argument("--only", choices=sorted(MIX),
                    help="draft a single category (--n is then that category's count)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8, help="excerpts per request")
    ap.add_argument("--sleep", type=float, default=13.0, help="seconds between requests")
    ap.add_argument("--exclude-source", nargs="*", default=list(NON_DOC_SOURCES),
                    help="pages to skip when sampling (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the per-category counts, call nothing")
    args = ap.parse_args()

    counts = plan_counts(args.n, args.only)
    unanswerable = sum(v for k, v in counts.items() if k.startswith("unanswerable"))
    total = sum(counts.values())
    _log("plan:")
    for qtype, m in counts.items():
        _log(f"  {qtype:28s} {m}")
    _log(f"  {'TOTAL':28s} {total}  ({unanswerable / total:.0%} unanswerable)")
    if args.dry_run:
        return

    chunks = load_chunks(args.chunks)
    eligible = eligible_chunk_ids(chunks, tuple(args.exclude_source))
    _log(f"loaded {len(chunks)} chunks; {len(eligible)} eligible after excluding "
         f"{', '.join(args.exclude_source) or 'nothing'}; drafting...")
    rows = generate_candidates(chunks, _load_default_llm(), counts, seed=args.seed,
                               batch=args.batch, pause=args.sleep,
                               exclude=tuple(args.exclude_source))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} candidates -> {out}")
    print("Now review them: python scripts/review_candidates.py")


if __name__ == "__main__":
    main()
