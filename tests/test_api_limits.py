"""Rate limiting and the /ask contract.

Every answered request spends money on a live API key, so these are the tests
that stand between a public demo and someone else's bill.
"""

import pytest
from fastapi.testclient import TestClient

import observable_rag.api.main as api
from observable_rag.api.limits import DAY, MINUTE, RateLimiter, hash_ip


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# --- the limiter ------------------------------------------------------------

def test_per_minute_limit_blocks_then_recovers():
    clock = _Clock()
    rl = RateLimiter(per_minute=3, per_day=100, global_per_day=1000, clock=clock)
    for _ in range(3):
        assert rl.check("ip").allowed
        rl.record("ip")

    blocked = rl.check("ip")
    assert not blocked.allowed and blocked.reason == "minute"
    assert blocked.retry_after > 0

    clock.advance(MINUTE + 1)
    assert rl.check("ip").allowed


def test_daily_limit_outlasts_the_minute_window():
    clock = _Clock()
    rl = RateLimiter(per_minute=100, per_day=5, global_per_day=1000, clock=clock)
    for _ in range(5):
        rl.record("ip")
    clock.advance(MINUTE * 10)          # minute window long gone
    assert rl.check("ip").reason == "daily"

    clock.advance(DAY)
    assert rl.check("ip").allowed


def test_one_visitor_cannot_exhaust_another():
    rl = RateLimiter(per_minute=2, per_day=100, global_per_day=1000, clock=_Clock())
    for _ in range(2):
        rl.record("noisy")
    assert not rl.check("noisy").allowed
    assert rl.check("someone-else").allowed


def test_global_cap_stops_everyone():
    """The per-IP limits stop one visitor; this is the one that bounds the bill."""
    rl = RateLimiter(per_minute=100, per_day=100, global_per_day=4, clock=_Clock())
    for i in range(4):
        rl.record(f"ip{i}")
    blocked = rl.check("a-brand-new-visitor")
    assert not blocked.allowed and blocked.reason == "busy"


def test_check_does_not_consume_quota():
    rl = RateLimiter(per_minute=1, per_day=10, global_per_day=10, clock=_Clock())
    for _ in range(5):
        assert rl.check("ip").allowed      # checking repeatedly must not spend it
    rl.record("ip")
    assert not rl.check("ip").allowed


def test_remaining_today_counts_down():
    rl = RateLimiter(per_minute=100, per_day=100, global_per_day=3, clock=_Clock())
    assert rl.remaining_today() == 3
    rl.record("ip")
    assert rl.remaining_today() == 2


def test_ip_hash_is_stable_and_not_the_address():
    h = hash_ip("203.0.113.7", salt="s")
    assert h == hash_ip("203.0.113.7", salt="s")
    assert h != hash_ip("203.0.113.8", salt="s")
    assert "203.0.113.7" not in h


# --- the endpoint -----------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """App with a fake pipeline, so nothing calls an LLM."""
    class _Fake:
        def answer(self, question):
            from observable_rag.generate.answer import RagOutput
            return RagOutput(answer=f"Answer [a#0] to {question}",
                             retrieved_chunk_ids=["a#0"], contexts=["ctx"],
                             citations=["a#0"], latency_ms=1.0)

    monkeypatch.setenv("EAGER_LOAD", "0")
    monkeypatch.setattr(api, "_pipeline", _Fake())
    monkeypatch.setattr(api, "_limiter",
                        RateLimiter(per_minute=2, per_day=10, global_per_day=10,
                                    clock=_Clock()))
    with TestClient(api.app) as c:
        yield c


def test_ask_answers_and_then_rate_limits(client):
    for _ in range(2):
        r = client.post("/ask", json={"question": "how do I use Depends?"})
        assert r.status_code == 200
        assert r.json()["citations"] == ["a#0"]

    r = client.post("/ask", json={"question": "again"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert "minute" in r.json()["detail"].lower()


def test_an_overlong_question_is_refused_before_it_costs_anything(client, monkeypatch):
    monkeypatch.setenv("MAX_QUESTION_LEN", "20")
    r = client.post("/ask", json={"question": "x" * 50})
    assert r.status_code == 413
    # and it did not consume quota
    assert client.post("/ask", json={"question": "short"}).status_code == 200


def test_an_empty_question_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_health_does_no_work(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_remaining_budget(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["requests_left_today"] == 10


def test_forwarded_header_identifies_the_visitor_behind_a_proxy(client):
    headers = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    for _ in range(2):
        assert client.post("/ask", json={"question": "q"},
                           headers=headers).status_code == 200
    assert client.post("/ask", json={"question": "q"},
                       headers=headers).status_code == 429
    # a different forwarded address is a different visitor
    assert client.post("/ask", json={"question": "q"},
                       headers={"x-forwarded-for": "203.0.113.10"}).status_code == 200
