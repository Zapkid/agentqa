"""Delegation policy: which tier a task starts at, and why (the 'why' is logged verbatim).

- deterministic_first: categories derivable from the spec never reach a model (Tier 0).
- static routing (cascade off): the profile's role -> tier mapping, no escalation.
- cascade: start at T1 unless difficulty >= threshold (then T2).
- learned routing (with cascade): Thompson-sample P(success) for T1 and T2 of this task type;
  start at T1 if its sample clears the target, else T2; with probability ``strong_tier_floor``
  start at T2 anyway (exploration floor). Seeded per task, so decisions are reproducible.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from agentqa.config import DispatchFile, Profile
from agentqa.dispatch.learned import RoutingStats

T1_TARGET = 0.6


@dataclass
class Decision:
    tier: str
    reason: str
    can_escalate: bool


def _rng(task_id: str) -> random.Random:
    return random.Random(int(hashlib.sha256(task_id.encode()).hexdigest()[:12], 16))  # noqa: S311 - bandit sampling, not cryptography


def choose_tier(
    task_id: str,
    task_type: str,
    difficulty: float,
    dispatch: DispatchFile,
    profile: Profile,
    stats: RoutingStats | None,
    forced_tier: str | None = None,
) -> Decision:
    if forced_tier is not None:
        return Decision(forced_tier, f"strategy forces {forced_tier}", can_escalate=False)
    if not dispatch.on("cascade"):
        tier = profile.tier_for_role("generator")
        return Decision(tier, f"static role routing: generator -> {tier}", can_escalate=False)
    threshold = dispatch.params.difficulty_t2_threshold
    if dispatch.on("learned_routing") and stats is not None:
        rng = _rng(task_id)
        if rng.random() < dispatch.params.strong_tier_floor:
            return Decision("T2", "exploration floor sampled T2", can_escalate=False)
        p1 = stats.sample(task_type, "T1", rng)
        # harder tasks need a more confident T1 to start cheap
        target = T1_TARGET + 0.3 * max(0.0, difficulty - 0.5)
        if p1 >= target:
            return Decision(
                "T1", f"thompson: P(T1 ok)~{p1:.2f} >= {target:.2f} (difficulty {difficulty})", True
            )
        return Decision(
            "T2", f"thompson: P(T1 ok)~{p1:.2f} < {target:.2f} (difficulty {difficulty})", False
        )
    if difficulty >= threshold:
        return Decision("T2", f"difficulty {difficulty} >= {threshold}", can_escalate=False)
    return Decision("T1", f"difficulty {difficulty} < {threshold}", can_escalate=True)
