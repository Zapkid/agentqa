"""Learned routing stats: per (task_type, tier) outcomes, used by Thompson sampling.

Priors come from the policy (pseudo-counts). Each run's verified outcomes update SQLite, so the
starting tier for a task type drifts towards the cheapest tier that keeps succeeding. A floor
keeps the strong tier sampled so the stats never go stale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentqa.store import Store

PRIORS = {"T1": (3.0, 2.0), "T2": (9.0, 1.0)}  # (alpha, beta) pseudo-counts: T1 ~0.6, T2 ~0.9


@dataclass
class Arm:
    successes: int = 0
    failures: int = 0
    tokens: int = 0

    def alpha_beta(self, tier: str) -> tuple[float, float]:
        a, b = PRIORS[tier]
        return a + self.successes, b + self.failures

    def mean(self, tier: str) -> float:
        a, b = self.alpha_beta(tier)
        return a / (a + b)


class RoutingStats:
    def __init__(self, store: Store) -> None:
        self.store = store

    def arm(self, task_type: str, tier: str) -> Arm:
        rows = self.store.query(
            "SELECT successes, failures, tokens FROM routing_stats WHERE task_type=? AND tier=?",
            (task_type, tier),
        )
        if not rows:
            return Arm()
        return Arm(rows[0]["successes"], rows[0]["failures"], rows[0]["tokens"])

    def update(self, task_type: str, tier: str, success: bool, tokens: int = 0) -> None:
        self.store.execute(
            "INSERT OR IGNORE INTO routing_stats (task_type, tier) VALUES (?, ?)", (task_type, tier)
        )
        col = "successes" if success else "failures"
        self.store.execute(
            f"UPDATE routing_stats SET {col} = {col} + 1, tokens = tokens + ? WHERE task_type=? AND tier=?",
            (tokens, task_type, tier),
        )

    def sample(self, task_type: str, tier: str, rng: random.Random) -> float:
        a, b = self.arm(task_type, tier).alpha_beta(tier)
        return rng.betavariate(a, b)

    def table(self) -> list[dict[str, object]]:
        out = []
        for r in self.store.query("SELECT * FROM routing_stats ORDER BY task_type, tier"):
            arm = Arm(r["successes"], r["failures"], r["tokens"])
            out.append(
                {
                    "task_type": r["task_type"],
                    "tier": r["tier"],
                    "successes": arm.successes,
                    "failures": arm.failures,
                    "posterior_mean": round(arm.mean(r["tier"]), 3),
                    "avg_tokens": round(arm.tokens / max(1, arm.successes + arm.failures)),
                }
            )
        return out
