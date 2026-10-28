"""Token usage and what it cost.

Every cost figure in this project used to be arithmetic over tiktoken counts
multiplied by a price someone typed from memory. That is an estimate wearing a
decimal point. Providers return the real token counts on every response, so
record those instead and multiply by prices kept in one editable file.

Prices change and are not discoverable from the API, so `data/pricing.json` is
the single place they live, and a model missing from it yields a null cost
rather than a quietly wrong number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Optional

PRICING_PATH = Path("data/pricing.json")


@dataclass
class Usage:
    """One call's tokens, and the cost if we know the price."""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def load_pricing(path: Path = PRICING_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def cost_usd(usage: Usage, pricing: Optional[dict] = None) -> Optional[float]:
    """Dollars for one call, or None if the model isn't priced.

    None rather than 0.0 on purpose: an unknown price is not a free call, and a
    report that sums unknowns as zero understates the bill silently.
    """
    prices = (pricing if pricing is not None else load_pricing()).get("models", {})
    price = prices.get(usage.model)
    if not price:
        return None
    return round(
        usage.prompt_tokens / 1_000_000 * price.get("input_per_1m", 0.0)
        + usage.completion_tokens / 1_000_000 * price.get("output_per_1m", 0.0),
        8)


@dataclass
class UsageRecorder:
    """Accumulates usage across calls. One per process; thread-safe because
    uvicorn serves sync handlers from a thread pool."""
    calls: list[Usage] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, usage: Usage) -> None:
        with self._lock:
            self.calls.append(usage)

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()

    def summary(self, pricing: Optional[dict] = None) -> dict:
        pricing = pricing if pricing is not None else load_pricing()
        with self._lock:
            calls = list(self.calls)
        costs = [c for c in (cost_usd(u, pricing) for u in calls) if c is not None]
        unpriced = sum(1 for u in calls if cost_usd(u, pricing) is None)
        return {
            "calls": len(calls),
            "prompt_tokens": sum(u.prompt_tokens for u in calls),
            "completion_tokens": sum(u.completion_tokens for u in calls),
            "total_tokens": sum(u.total_tokens for u in calls),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "unpriced_calls": unpriced,
            "models": sorted({u.model for u in calls}),
        }


def extract_usage(response, model: str) -> Usage:
    """Pull token counts off an OpenAI-shaped response. Missing counts are 0, so
    a provider that doesn't report them shows zero tokens rather than crashing."""
    u = getattr(response, "usage", None)
    return Usage(
        model=model,
        prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
    )
