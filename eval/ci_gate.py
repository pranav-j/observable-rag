"""CI gate: fail a pull request if quality regresses against the baseline.

Compares the freshly produced eval report against a committed baseline using
RELATIVE thresholds -- "this PR must not drop recall by more than 3 points"
rather than an absolute "recall must exceed 95%". Absolute gates on noisy
LLM-judged metrics either never pass or get quietly tuned around; relative
regression detection is what actually catches a change that made things worse.

Usage:
    python eval/ci_gate.py --baseline data/eval/baseline.json --current eval_report.json

Exit code 0 = pass, 1 = a gated metric regressed past its margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# How far each metric may move in the WORSE direction before the gate fails.
#   higher_is_better=True  -> a drop of more than `margin` fails
#   higher_is_better=False -> a rise of more than `margin` fails
# Margins are sized against the golden set's denominators -- 78 answerable and
# 26 unanswerable items -- so each one is expressed as "how many questions may
# regress before this fails". See docs/gate-margins.md for the derivation.
#
#   retrieval metrics are deterministic given a fixed index, so their only
#   run-to-run movement is corpus drift; they stay tight (~2 questions).
#   judge metrics vary with regeneration, not with the judge itself (replaying
#   it at temperature 0 measured a standard deviation of exactly 0), so their
#   margins are a stated placeholder until repeated full runs measure it.
# Judge and robustness margins are 3x the standard deviation measured across
# three full runs of identical code (reports/eval_run{1,2,3}.json), rounded to
# the nearest question. Retrieval measured stdev 0.0000 across those runs -- it
# is deterministic given a fixed index -- so those stay tight and absorb about
# two questions of 78, which is drift, not noise.
GATES = {
    ("retrieval", "recall_at_5"):             {"margin": 0.03, "higher_is_better": True},
    ("retrieval", "all_relevant_at_5"):       {"margin": 0.03, "higher_is_better": True},
    ("retrieval", "context_precision_at_5"):  {"margin": 0.03, "higher_is_better": True},
    ("retrieval", "mrr"):                     {"margin": 0.03, "higher_is_better": True},
    ("generation", "faithfulness"):           {"margin": 0.03, "higher_is_better": True},
    ("generation", "answer_relevance"):       {"margin": 0.02, "higher_is_better": True},
    # The noisiest metric by a distance: stdev 0.0336, spread 0.071 across three
    # identical runs. 0.05 was tighter than its own noise floor and would have
    # failed on wobble, exactly as the 150ms latency margin did.
    ("generation", "citation_precision"):     {"margin": 0.10, "higher_is_better": True},
    ("generation", "citation_validity"):      {"margin": 0.02, "higher_is_better": True},
    ("robustness", "false_answer_rate"):      {"margin": 0.08, "higher_is_better": False},
    ("robustness", "false_abstention_rate"):  {"margin": 0.04, "higher_is_better": False},
    # Declined and then answered anyway, on a question the sources answer.
    ("robustness", "contradiction_rate"):     {"margin": 0.03, "higher_is_better": False},
    # End-to-end p95 is mostly the provider's latency, which no pull request
    # controls: four runs of near-identical code spanned 30.5s to 43.3s. The old
    # 150ms margin failed on that noise every time. 15s still catches a genuine
    # doubling while absorbing the spread we have actually observed. Replace this
    # with retrieval-stage latency once the pipeline emits per-stage spans.
    ("latency_ms", "p95"):                    {"margin": 15000.0, "higher_is_better": False},
}


def _get(report: dict, section: str, name: str):
    return report.get(section, {}).get(name)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate a PR on relative metric regression.")
    ap.add_argument("--baseline", type=Path, default=Path("data/eval/baseline.json"))
    ap.add_argument("--current", type=Path, default=Path("eval_report.json"))
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))

    failures: list[str] = []
    rows: list[tuple] = []

    for (section, name), rule in GATES.items():
        base = _get(baseline, section, name)
        curr = _get(current, section, name)

        # A metric the baseline measured but this run didn't is a FAILURE, not a
        # skip. The judge swallows its own errors and returns None, the harness
        # drops Nones when averaging, and treating the resulting null as "skip"
        # let a run where the judge produced nothing sail through green.
        if base is not None and curr is None:
            rows.append((f"{section}.{name}", base, curr, "—", "MISSING"))
            failures.append(
                f"{section}.{name} is missing from this run but the baseline has it "
                f"({base}). The metric was not measured -- check the judge and its "
                f"credentials, or gate a run that measured it.")
            continue

        if base is None or curr is None:
            rows.append((f"{section}.{name}", base, curr, "—", "skip"))
            continue

        delta = curr - base
        regressed = (delta < -rule["margin"]) if rule["higher_is_better"] \
            else (delta > rule["margin"])
        rows.append((f"{section}.{name}", base, curr, f"{delta:+.3f}",
                     "FAIL" if regressed else "ok"))
        if regressed:
            direction = "dropped" if rule["higher_is_better"] else "rose"
            failures.append(
                f"{section}.{name} {direction} by {abs(delta):.3f} "
                f"(margin {rule['margin']}): {base} -> {curr}")

    width = max(len(r[0]) for r in rows)
    print(f"{'metric'.ljust(width)}  {'base':>8}  {'curr':>8}  {'delta':>8}  status")
    for name, base, curr, delta, status in rows:
        b = "—".rjust(8) if base is None else f"{base:8.3f}"
        c = "—".rjust(8) if curr is None else f"{curr:8.3f}"
        print(f"{name.ljust(width)}  {b}  {c}  {delta:>8}  {status}")

    if failures:
        print("\nQuality gate FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())