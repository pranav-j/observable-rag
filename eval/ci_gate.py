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
GATES = {
    ("retrieval", "recall_at_5"):            {"margin": 0.03, "higher_is_better": True},
    ("retrieval", "context_precision_at_5"): {"margin": 0.03, "higher_is_better": True},
    ("generation", "faithfulness"):          {"margin": 0.03, "higher_is_better": True},
    ("generation", "answer_relevance"):      {"margin": 0.03, "higher_is_better": True},
    ("robustness", "false_answer_rate"):     {"margin": 0.02, "higher_is_better": False},
    ("latency_ms", "p95"):                   {"margin": 150.0, "higher_is_better": False},
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
