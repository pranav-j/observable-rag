"""Check the LLM judge against human labels.

faithfulness and citation_precision are produced by a model. Until someone
checks that model against human judgement, they're just a second opinion with a
decimal point on it. This draws a sample of real answers, lets you label them,
and reports how often the judge agreed.

    python eval/evaluate.py                        # produces eval_items.jsonl
    python scripts/calibrate_judge.py sample       # -> judge_calibration.jsonl
    python scripts/calibrate_judge.py label        # you label them
    python scripts/calibrate_judge.py report       # agreement + Cohen's kappa

The sample is drawn by RISK, assessed from retrieval and citation facts and
never from the judge's own score. Stratifying on the judge was tried first and
failed: it scored 39 of 40 sampled answers at exactly 1.0, so the sample was
thirty-nine foregone agreements, and kappa over a near-constant rater is ~0
whatever the human says.

Only you set the human labels. Nothing here writes one on your behalf, and the
judge's scores stay hidden until after you decide -- an opinion shown first is
an opinion you partly adopt, and agreement is the thing being measured.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ITEMS = Path("eval_items.jsonl")
CALIBRATION = Path("data/eval/judge_calibration.jsonl")
DEFAULT_THRESHOLD = 0.5

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")


# --- agreement maths --------------------------------------------------------

def percent_agreement(a: list[int], b: list[int]) -> "float | None":
    if not a:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: list[int], b: list[int]) -> "float | None":
    """Chance-corrected agreement for two binary raters.

    Returns None only when the statistic genuinely isn't defined: no items, or
    both raters gave the same single label to everything, which makes expected
    agreement 1 and the denominator 0.

    When just ONE rater is constant -- a judge that calls every answer supported
    -- kappa is defined and comes out at 0.0. That is the honest answer: perfect
    raw agreement on the majority class, no skill beyond chance. It's the number
    to expect from an uncalibrated judge on a set where almost everything is in
    fact supported.
    """
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in (0, 1))
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def interpret_kappa(k: "float | None") -> str:
    """Landis & Koch bands, with an explicit label for the degenerate zero."""
    if k is None:
        return "undefined (both raters used one label throughout)"
    for threshold, label in ((0.81, "almost perfect"), (0.61, "substantial"),
                             (0.41, "moderate"), (0.21, "fair"), (0.01, "slight")):
        if k >= threshold:
            return label
    if k > -0.01:
        return "none -- no better than chance"
    return "worse than chance"


def _binarise(score: "float | None", threshold: float) -> "int | None":
    return None if score is None else int(score >= threshold)


# --- sampling ---------------------------------------------------------------

def _score_band(score: "float | None") -> str:
    if score is None:
        return "ungraded"
    return "high" if score >= 0.9 else ("mid" if score >= 0.5 else "low")


def risk_reasons(item: dict) -> list[str]:
    """Why this answer might be unfaithful, judged WITHOUT the judge.

    Stratifying on the judge's own score failed: it scored 39 of 40 sampled
    answers at exactly 1.0, so the sample was 39 near-certain agreements and
    kappa over a near-constant rater is ~0 no matter what the human says.

    These signals are independent of the judge, so a sample built from them can
    actually contain disagreement:

      retrieval_missed  the labelled chunk never made the top k, so whatever
                        the answer is built on, it isn't the ground truth
      answered_unanswerable  the docs don't cover it and it answered anyway
      near_miss         the question is adjacent to the chunk, not answered by it
      invalid_citations the model emitted a tag that resolves to nothing
      multi_citation    more citations, more chances one is unsupported
    """
    reasons = []
    if item.get("answerable") and item.get("recall") == 0.0:
        reasons.append("retrieval_missed")
    if not item.get("answerable") and not item.get("abstained"):
        reasons.append("answered_unanswerable")
    if item.get("qtype") == "near_miss":
        reasons.append("near_miss")
    if item.get("invalid_citations"):
        reasons.append("invalid_citations")
    if len(item.get("citations") or []) > 1:
        reasons.append("multi_citation")
    return reasons


def stratified_sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Risky answers first, then a spread across categories.

    Risk is assessed from retrieval and citation facts, never from the judge's
    score -- seeding the sample with the judge's opinion is how you end up
    measuring whether a human agrees with the cases the judge already found
    easy.
    """
    graded = [it for it in items if not it.get("abstained") and it.get("answer", "").strip()]
    # unanswerable items that got answered aren't "graded" by the judge but are
    # exactly the ones worth a human read
    graded += [it for it in items
               if not it.get("answerable") and not it.get("abstained")
               and it not in graded]

    rng = random.Random(seed)
    rng.shuffle(graded)

    risky = [it for it in graded if risk_reasons(it)]
    risky.sort(key=lambda it: -len(risk_reasons(it)))
    picked = risky[:n]

    # fill the rest round-robin across categories so no one qtype dominates
    if len(picked) < n:
        rest: dict[str, list[dict]] = defaultdict(list)
        for it in graded:
            if it not in picked:
                rest[it.get("qtype", "factual")].append(it)
        keys = sorted(rest)
        while len(picked) < n and any(rest[k] for k in keys):
            for k in keys:
                if rest[k] and len(picked) < n:
                    picked.append(rest[k].pop())
    return picked


def build_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    rows = []
    for it in stratified_sample(items, n, seed):
        rows.append({
            "id": it["id"],
            "question": it["question"],
            "answer": it["answer"],
            "contexts": it.get("contexts", []),
            "citations": it.get("citations", []),
            "qtype": it.get("qtype"),
            "answerable": it.get("answerable", True),
            "risk": risk_reasons(it),
            "judge_faithfulness": it.get("faithfulness"),
            "judge_citation_precision": it.get("citation_precision"),
            # filled in by you, never by the tool
            "human_faithful": None,
            "human_citations_correct": None,
        })
    return rows


# --- labelling --------------------------------------------------------------

def _wrap(text: str, width: int = 88) -> str:
    out = []
    for para in text.split("\n"):
        while len(para) > width:
            cut = para.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            out.append(para[:cut])
            para = para[cut:].lstrip()
        out.append(para)
    return "\n".join(out)


def _ask(prompt: str) -> "bool | None":
    """y / n / s(kip). Returns None to leave the label unset."""
    while True:
        key = input(f"  {prompt} {BOLD}[y/n/s]{RESET} ").strip().lower()
        if key in ("y", "yes"):
            return True
        if key in ("n", "no"):
            return False
        if key in ("s", "skip", ""):
            return None
        print(f"{DIM}  y, n, or s{RESET}")


def label(rows: list[dict], path: Path) -> int:
    todo = [r for r in rows if r["human_faithful"] is None]
    if not todo:
        print("every row is already labelled")
        return 0
    print(f"{len(todo)} to label. The judge's own scores are hidden until you've "
          f"decided, so they can't anchor you.\n")

    done = 0
    for i, row in enumerate(todo, start=1):
        print(f"\n{BOLD}{'─' * 88}{RESET}")
        why = ", ".join(row.get("risk") or []) or "sampled for coverage"
        print(f"{BOLD}[{i}/{len(todo)}]{RESET} {DIM}{row['id']}{RESET} "
              f"{YELLOW}{row['qtype']}{RESET}  {DIM}({why}){RESET}")
        print(f"\n{BOLD}Q:{RESET} {_wrap(row['question'])}")
        print(f"\n{BOLD}ANSWER:{RESET}\n{_wrap(row['answer'])}")
        print(f"\n{BOLD}SOURCES SHOWN TO THE MODEL:{RESET}")
        for cid, ctx in zip(row.get("retrieved_chunk_ids", row["citations"]),
                            row["contexts"]):
            print(f"\n{DIM}── {cid}{RESET}\n{_wrap(ctx[:900])}")

        try:
            row["human_faithful"] = _ask("Is every claim supported by those sources?")
            if row["citations"]:
                row["human_citations_correct"] = _ask(
                    "Does each citation back the sentence it's on?")
        except (EOFError, KeyboardInterrupt):
            print("\ninterrupted; labels so far are saved")
            break

        print(f"{DIM}  judge said: faithfulness="
              f"{row['judge_faithfulness']}, citations="
              f"{row['judge_citation_precision']}{RESET}")
        done += 1
        _write(path, rows)

    print(f"\nlabelled {done}")
    return 0


# --- reporting --------------------------------------------------------------

def agreement_report(rows: list[dict], threshold: float) -> dict:
    report: dict = {"threshold": threshold, "n_sampled": len(rows), "metrics": {}}
    pairs = {
        "faithfulness": ("judge_faithfulness", "human_faithful"),
        "citation_precision": ("judge_citation_precision", "human_citations_correct"),
    }
    for name, (judge_key, human_key) in pairs.items():
        judge, human = [], []
        for row in rows:
            h, j = row.get(human_key), _binarise(row.get(judge_key), threshold)
            if h is None or j is None:
                continue
            human.append(int(h))
            judge.append(j)
        kappa = cohens_kappa(judge, human)
        report["metrics"][name] = {
            "n_labelled": len(judge),
            "percent_agreement": round(percent_agreement(judge, human), 4)
            if judge else None,
            "cohens_kappa": round(kappa, 4) if kappa is not None else None,
            "interpretation": interpret_kappa(kappa) if judge else "no labels yet",
            "judge_positive_rate": round(sum(judge) / len(judge), 4) if judge else None,
            "human_positive_rate": round(sum(human) / len(human), 4) if human else None,
        }
    return report


def print_report(report: dict) -> None:
    print(f"\n{BOLD}Judge vs human{RESET}  "
          f"{DIM}(judge binarised at >= {report['threshold']}){RESET}")
    for name, m in report["metrics"].items():
        print(f"\n  {BOLD}{name}{RESET}")
        if not m["n_labelled"]:
            print(f"    {DIM}no labels yet{RESET}")
            continue
        print(f"    labelled pairs      {m['n_labelled']}")
        print(f"    percent agreement   {m['percent_agreement']:.1%}")
        kappa = m["cohens_kappa"]
        colour = GREEN if (kappa or 0) >= 0.61 else (YELLOW if (kappa or 0) >= 0.41 else RED)
        shown = f"{kappa:+.3f}" if kappa is not None else "—"
        print(f"    Cohen's kappa       {colour}{shown}{RESET} ({m['interpretation']})")
        print(f"    positive rate       judge {m['judge_positive_rate']:.0%}, "
              f"human {m['human_positive_rate']:.0%}")


# --- io ---------------------------------------------------------------------

def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the LLM judge against human labels.")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="draw a stratified sample to label")
    s.add_argument("--items", type=Path, default=ITEMS)
    s.add_argument("--out", type=Path, default=CALIBRATION)
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)

    label_parser = sub.add_parser("label", help="label the sample")
    label_parser.add_argument("--file", type=Path, default=CALIBRATION)

    r = sub.add_parser("report", help="judge-vs-human agreement")
    r.add_argument("--file", type=Path, default=CALIBRATION)
    r.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    r.add_argument("--json", action="store_true")

    args = ap.parse_args()

    if args.command == "sample":
        items = _read(args.items)
        if not items:
            print(f"no per-item results at {args.items}; run eval/evaluate.py first")
            return 1
        rows = build_sample(items, args.n, args.seed)
        _write(args.out, rows)
        bands = Counter(_score_band(r["judge_faithfulness"]) for r in rows)
        types = Counter(r["qtype"] for r in rows)
        print(f"wrote {len(rows)} rows -> {args.out}")
        print(f"  by judge score: {dict(bands)}")
        print(f"  by qtype:       {dict(types)}")
        print(f"\nNow label them: python {sys.argv[0]} label")
        return 0

    rows = _read(args.file)
    if not rows:
        print(f"nothing at {args.file}; run `sample` first")
        return 1

    if args.command == "label":
        return label(rows, args.file)

    report = agreement_report(rows, args.threshold)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
