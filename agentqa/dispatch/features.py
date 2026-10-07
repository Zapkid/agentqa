"""Transparent difficulty features for a generation task (no learned black box).

difficulty = 0.15 + 0.25*stateful + 0.2*multi_role + 0.1*(risk>=4) + 0.15*money_or_crypto
             + 0.05*min(schema_depth,3) + 0.1*prior_failures_on_endpoint (capped at 1.0)
"""

from __future__ import annotations

from typing import Any

from agentqa.models import Endpoint, TestIntent

STATEFUL = {"state-transition", "idempotency", "webhooks/signatures", "time/timezone"}
MONEY_OR_CRYPTO = {"money/precision", "webhooks/signatures"}
MULTI_ROLE = {"auth/permission", "data-leak"}


def schema_depth(schema: dict[str, Any] | None, depth: int = 0) -> int:
    if not schema or depth > 6:
        return depth
    children = list((schema.get("properties") or {}).values())
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        children.append(schema["items"])
    return max([depth] + [schema_depth(c, depth + 1) for c in children if isinstance(c, dict)])


def features(intent: TestIntent, ep: Endpoint | None, prior_failures: int = 0) -> dict[str, Any]:
    return {
        "stateful": intent.category in STATEFUL,
        "multi_role": intent.category in MULTI_ROLE,
        "high_risk": intent.risk >= 4,
        "money_or_crypto": intent.category in MONEY_OR_CRYPTO,
        "schema_depth": schema_depth(ep.request_schema) if ep else 0,
        "preconditions": len(intent.preconditions),
        "prior_failures": prior_failures,
    }


def difficulty(f: dict[str, Any]) -> float:
    score = (
        0.15
        + 0.25 * f["stateful"]
        + 0.2 * f["multi_role"]
        + 0.1 * f["high_risk"]
        + 0.15 * f["money_or_crypto"]
        + 0.05 * min(int(f["schema_depth"]), 3)
        + 0.1 * min(int(f["prior_failures"]), 3)
    )
    return round(min(1.0, score), 3)
