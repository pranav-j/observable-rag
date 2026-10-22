"""Spend and abuse protection for a publicly reachable /ask.

Every answered request costs money on someone's API key, so a public endpoint
without caps is an open invitation to spend it. Four limits, all configurable by
environment variable:

    RATE_PER_MINUTE   per-IP requests per minute        (default 5)
    RATE_PER_DAY      per-IP requests per day           (default 30)
    GLOBAL_PER_DAY    total requests per day, all IPs   (default 500)
    MAX_QUESTION_LEN  characters                        (default 500)

GLOBAL_PER_DAY is the one that actually bounds the bill; the per-IP limits just
stop one visitor consuming it. Set it below whatever you are willing to pay for
in a day.

State is in-process and resets when the container restarts. That is the right
shape here because embedded Qdrant already holds an exclusive file lock, so
there is exactly one process serving; a second instance would need Redis and a
different design.

IP addresses are hashed before they are logged or stored. The limiter needs to
tell visitors apart, not to know who they are.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

MINUTE = 60.0
DAY = 86_400.0


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name) or default))
    except ValueError:
        return default


def hash_ip(ip: str, salt: str = "") -> str:
    """Short, stable, non-reversible handle for an address."""
    salt = salt or os.getenv("IP_HASH_SALT", "observable-rag")
    return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:12]


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: int = 0


class RateLimiter:
    """Sliding-window limiter, per hashed IP plus a global daily cap."""

    def __init__(self, per_minute: int | None = None, per_day: int | None = None,
                 global_per_day: int | None = None, clock=time.monotonic):
        self.per_minute = _env_int("RATE_PER_MINUTE", 5) if per_minute is None else per_minute
        self.per_day = _env_int("RATE_PER_DAY", 30) if per_day is None else per_day
        self.global_per_day = (_env_int("GLOBAL_PER_DAY", 500)
                               if global_per_day is None else global_per_day)
        self._clock = clock
        self._per_ip: dict[str, deque] = defaultdict(deque)
        self._global: deque = deque()

    @staticmethod
    def _prune(hits: deque, now: float, window: float) -> None:
        while hits and now - hits[0] > window:
            hits.popleft()

    def check(self, ip_hash: str) -> Decision:
        """Would this request be allowed? Does not consume quota."""
        now = self._clock()
        self._prune(self._global, now, DAY)
        if self.global_per_day and len(self._global) >= self.global_per_day:
            oldest = self._global[0] if self._global else now
            return Decision(False, "busy", int(DAY - (now - oldest)) + 1)

        hits = self._per_ip[ip_hash]
        self._prune(hits, now, DAY)
        if self.per_day and len(hits) >= self.per_day:
            return Decision(False, "daily", int(DAY - (now - hits[0])) + 1)

        recent = sum(1 for t in hits if now - t <= MINUTE)
        if self.per_minute and recent >= self.per_minute:
            first_recent = next(t for t in hits if now - t <= MINUTE)
            return Decision(False, "minute", int(MINUTE - (now - first_recent)) + 1)

        return Decision(True)

    def record(self, ip_hash: str) -> None:
        """Consume one unit of quota. Call only for requests that cost money."""
        now = self._clock()
        self._per_ip[ip_hash].append(now)
        self._global.append(now)

    def remaining_today(self) -> int:
        self._prune(self._global, self._clock(), DAY)
        return max(0, self.global_per_day - len(self._global)) if self.global_per_day else -1


MESSAGES = {
    "minute": "That's a few questions in quick succession — give it a minute.",
    "daily": "You've hit the daily limit for this demo. It resets in a day.",
    "busy": "The demo has hit its daily budget. It resets in a day — the code and "
            "the eval numbers are all in the repo in the meantime.",
}


def too_long_message(limit: int) -> str:
    return f"Questions are capped at {limit} characters for this demo."


def max_question_len() -> int:
    return _env_int("MAX_QUESTION_LEN", 500)
