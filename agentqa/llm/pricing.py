"""Cost ledger arithmetic: actual cost (0 on free tiers) and list-equivalent cost."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from agentqa import config
from agentqa.llm.types import Usage


def list_cost(model: str, usage: Usage) -> float:
    price = config.pricing().models.get(model)
    if price is None:
        raise KeyError(f"no price for model {model!r} in config/pricing.yaml")
    uncached = max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens)
    return (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.cache_write_tokens * price.cache_write
        + usage.output_tokens * price.output
    ) / 1_000_000


def costs(provider: str, model: str, usage: Usage) -> tuple[float, float]:
    """Return (cost_usd_actual, cost_usd_list_equivalent)."""
    list_eq = list_cost(model, usage)
    free = config.providers().providers[provider].free_tier
    return (0.0 if free else list_eq), list_eq


@dataclass
class LedgerEntry:
    agent: str
    provider: str
    model: str
    tier: str | None
    usage: Usage
    cost_usd_actual: float
    cost_usd_list_equivalent: float
    cache_status: str
    task_id: str | None = None


@dataclass
class CostLedger:
    """Per-run accumulation of every LLM call's tokens and cost."""

    entries: list[LedgerEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, entry: LedgerEntry) -> None:
        with self._lock:
            self.entries.append(entry)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for e in self.entries:
            total = total + e.usage
        return total

    @property
    def cost_actual(self) -> float:
        return sum(e.cost_usd_actual for e in self.entries)

    @property
    def cost_list(self) -> float:
        return sum(e.cost_usd_list_equivalent for e in self.entries)

    def by(self, attr: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for e in self.entries:
            key = str(getattr(e, attr))
            row = out.setdefault(
                key,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "cost_usd_actual": 0.0,
                    "cost_usd_list_equivalent": 0.0,
                },
            )
            row["calls"] += 1
            row["input_tokens"] += e.usage.input_tokens
            row["output_tokens"] += e.usage.output_tokens
            row["cached_input_tokens"] += e.usage.cached_input_tokens
            row["cost_usd_actual"] += e.cost_usd_actual
            row["cost_usd_list_equivalent"] += e.cost_usd_list_equivalent
        return out

    def summary(self) -> dict[str, object]:
        u = self.usage
        return {
            "calls": len(self.entries),
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cached_input_tokens": u.cached_input_tokens,
            "total_tokens": u.total,
            "cost_usd_actual": round(self.cost_actual, 6),
            "cost_usd_list_equivalent": round(self.cost_list, 6),
            "by_agent": self.by("agent"),
            "by_model": self.by("model"),
            "by_tier": self.by("tier"),
        }
