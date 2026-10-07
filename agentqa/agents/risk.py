"""Transparent risk score (1-5) from data sensitivity, money movement and blast radius.

The planner model proposes a risk; ``clamp_risk`` keeps it within one point of this rule-based
score, so a model cannot silently push a money-movement test to the bottom of the queue.
"""

from __future__ import annotations

from agentqa.models import Endpoint

MONEY_WORDS = ("order", "invoice", "payment", "price", "total", "amount", "refund", "checkout")
SENSITIVE_WORDS = ("customer", "user", "account", "email", "invoice", "payment", "profile")
HIGH_IMPACT_CATEGORIES = {"auth/permission", "webhooks/signatures", "money/precision", "data-leak"}


def rule_risk(ep: Endpoint, category: str) -> int:
    text = f"{ep.path} {' '.join(ep.tags)} {ep.summary}".lower()
    score = 1
    if any(w in text for w in MONEY_WORDS):
        score += 1
    if any(w in text for w in SENSITIVE_WORDS):
        score += 1
    if ep.method in ("POST", "PUT", "PATCH", "DELETE"):
        score += 1
    if category in HIGH_IMPACT_CATEGORIES:
        score += 1
    return max(1, min(5, score))


def clamp_risk(proposed: int, ep: Endpoint, category: str) -> int:
    base = rule_risk(ep, category)
    return max(1, min(5, max(base - 1, min(base + 1, proposed))))
