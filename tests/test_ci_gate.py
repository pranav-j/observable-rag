"""CI gate behavior: regressions fail (exit 1), noise within margin passes (exit 0)."""

import json
import subprocess
import sys

GATE = "eval/ci_gate.py"
BASELINE = {
    "retrieval": {"recall_at_5": 0.90, "context_precision_at_5": 0.19, "mrr": 0.78},
    "generation": {"faithfulness": 0.99, "answer_relevance": 0.98},
    "robustness": {"false_answer_rate": 0.0, "over_abstain_rate": 0.0},
    "latency_ms": {"p50": 14000.0, "p95": 34000.0},
}


def _run(tmp_path, current):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    b.write_text(json.dumps(BASELINE))
    c.write_text(json.dumps(current))
    return subprocess.run(
        [sys.executable, GATE, "--baseline", str(b), "--current", str(c)],
        capture_output=True, text=True)


def _mutate(**path_value):
    import copy
    cur = copy.deepcopy(BASELINE)
    for dotted, value in path_value.items():
        section, name = dotted.split("__")
        cur[section][name] = value
    return cur


def test_gate_passes_when_unchanged(tmp_path):
    r = _run(tmp_path, BASELINE)
    assert r.returncode == 0, r.stdout


def test_gate_fails_on_recall_regression(tmp_path):
    r = _run(tmp_path, _mutate(retrieval__recall_at_5=0.80))  # -0.10, past 0.03
    assert r.returncode == 1
    assert "recall_at_5" in r.stdout


def test_gate_fails_when_false_answers_rise(tmp_path):
    r = _run(tmp_path, _mutate(robustness__false_answer_rate=0.20))  # past 0.02
    assert r.returncode == 1


def test_gate_tolerates_noise_within_margin(tmp_path):
    r = _run(tmp_path, _mutate(generation__faithfulness=0.97))  # -0.02, within 0.03
    assert r.returncode == 0, r.stdout