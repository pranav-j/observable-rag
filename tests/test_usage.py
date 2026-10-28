"""Token accounting and cost.

Every cost figure in this project used to be tiktoken arithmetic times a price
typed from memory. These cover the parts that turn that into a measurement.
"""

import observable_rag.generate.llm as llm
from observable_rag.generate.usage import (
    Usage,
    UsageRecorder,
    cost_usd,
    extract_usage,
)

PRICING = {"models": {
    "cheap": {"input_per_1m": 1.0, "output_per_1m": 2.0},
}}


def test_cost_uses_separate_input_and_output_prices():
    u = Usage(model="cheap", prompt_tokens=1_000_000, completion_tokens=500_000)
    assert cost_usd(u, PRICING) == 2.0   # 1.00 in + 1.00 out


def test_an_unpriced_model_costs_none_not_zero():
    """Summing unknown prices as zero understates the bill silently."""
    assert cost_usd(Usage(model="mystery", prompt_tokens=10**6), PRICING) is None


def test_recorder_totals_and_flags_unpriced_calls():
    r = UsageRecorder()
    r.record(Usage("cheap", prompt_tokens=1_000_000, completion_tokens=0))
    r.record(Usage("mystery", prompt_tokens=2_000, completion_tokens=1_000))
    s = r.summary(PRICING)
    assert s["calls"] == 2
    assert s["total_tokens"] == 1_003_000
    assert s["cost_usd"] == 1.0          # only the priced call contributes
    assert s["unpriced_calls"] == 1      # and the gap is visible
    assert s["models"] == ["cheap", "mystery"]


def test_summary_of_nothing_is_not_zero_cost():
    assert UsageRecorder().summary(PRICING)["cost_usd"] is None


def test_recorder_resets():
    r = UsageRecorder()
    r.record(Usage("cheap", prompt_tokens=10))
    r.reset()
    assert r.summary(PRICING)["calls"] == 0


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens = p, c


class _Resp:
    def __init__(self, p=0, c=0, usage=True):
        self.usage = _Usage(p, c) if usage else None
        self.choices = [type("C", (), {"message": type("M", (), {"content": "hi"})()})()]


def test_extract_reads_the_providers_counts():
    u = extract_usage(_Resp(120, 30), "cheap")
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (120, 30, 150)


def test_a_provider_that_reports_no_usage_does_not_crash():
    u = extract_usage(_Resp(usage=False), "cheap")
    assert u.total_tokens == 0


def test_the_completer_records_what_the_provider_billed():
    """The counts come off the response, not from counting tokens locally."""
    class _Completions:
        def create(self, **kw):
            return _Resp(200, 50)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    rec = UsageRecorder()
    complete = llm._chat_completer(_Client(), "cheap", rec)
    complete([{"role": "user", "content": "hi"}])

    s = rec.summary(PRICING)
    assert s["calls"] == 1
    assert s["prompt_tokens"] == 200 and s["completion_tokens"] == 50
    assert s["cost_usd"] == round(200 / 1e6 * 1.0 + 50 / 1e6 * 2.0, 6)


def test_no_recorder_means_no_bookkeeping():
    class _Completions:
        def create(self, **kw):
            return _Resp(1, 1)

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    assert llm._chat_completer(_Client(), "cheap")([{"role": "user", "content": "x"}]) == "hi"
