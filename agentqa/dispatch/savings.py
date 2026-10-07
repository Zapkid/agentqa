"""Cost-saving mechanisms that act on the set of intents (each toggled in config/dispatch.yaml).

- dedupe: drop near-identical intents (embedding similarity) before generation
- budget allocation: spend where the risk is; cheapest path or skip-with-reason when tight
- incremental: reuse plans and validated tests for endpoints whose spec and docs are unchanged
- batching: group small, similar intents into one generator call (per-item validation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentqa.ingest.embed import cosine, default_embedder
from agentqa.models import TestIntent, ValidatedTest
from agentqa.obs import metrics, tracing
from agentqa.store import Store

# Prior token estimates per generation task (used before learned stats exist).
EST_TOKENS = {"T1": 3500, "T2": 3500}


def dedupe(
    intents: list[TestIntent], threshold: float
) -> tuple[list[TestIntent], list[TestIntent]]:
    """Keep the first (highest-risk) of each group of near-duplicates. Returns (kept, removed)."""
    ordered = sorted(intents, key=lambda i: -i.risk)
    texts = [f"{i.endpoint} {i.category} {i.title} {i.expected_behavior}" for i in ordered]
    vectors = default_embedder().embed(texts)
    kept: list[tuple[TestIntent, list[float]]] = []
    removed: list[TestIntent] = []
    for intent, vec in zip(ordered, vectors, strict=True):
        dup = next(
            (k for k, kv in kept if k.endpoint == intent.endpoint and cosine(vec, kv) >= threshold),
            None,
        )
        if dup is not None:
            removed.append(intent)
            tracing.event("dedupe.removed", intent=intent.id, duplicate_of=dup.id)
        else:
            kept.append((intent, vec))
    if removed:
        metrics.inc("agentqa_dedupe_removed_total", len(removed))
    keep_ids = {k.id for k, _ in kept}
    return [i for i in intents if i.id in keep_ids], removed


@dataclass
class Allocation:
    tier_cap: dict[str, str]  # intent_id -> max tier allowed ("T1" means no escalation)
    skipped: dict[str, str]  # intent_id -> reason


def allocate(
    intents: list[TestIntent], tokens_available: int, est: dict[str, int] | None = None
) -> Allocation:
    """Risk-weighted allocation. Highest risk first gets full cascade (T1 + escalation to T2,
    ~2 calls' worth); when the budget runs short, lower-risk intents are capped at T1 without
    escalation, and finally skipped and listed as uncovered with the reason."""
    est = est or EST_TOKENS
    full, cheap = est["T1"] + est["T2"] // 2, est["T1"]
    remaining = tokens_available
    cap: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for intent in sorted(intents, key=lambda i: (-i.risk, i.id)):
        if remaining >= full:
            cap[intent.id] = "T2"
            remaining -= full
        elif remaining >= cheap:
            cap[intent.id] = "T1"
            remaining -= cheap
            tracing.event("budget.downgraded", intent=intent.id, risk=intent.risk)
        else:
            skipped[intent.id] = (
                f"budget: risk {intent.risk} intent skipped to stay within the run budget"
            )
            tracing.event("budget.skipped", intent=intent.id, risk=intent.risk)
    return Allocation(cap, skipped)


class Incremental:
    """Per-endpoint artifacts keyed by the endpoint's spec hash and the docs digest."""

    def __init__(self, store: Store, api_key: str) -> None:
        self.store = store
        self.api_key = api_key

    def load(self, endpoint: str, endpoint_hash: str) -> dict[str, Any] | None:
        rows = self.store.query(
            "SELECT endpoint_hash, payload FROM artifacts WHERE api_key=? AND endpoint=? AND kind='plan'",
            (self.api_key, endpoint),
        )
        if not rows or rows[0]["endpoint_hash"] != endpoint_hash:
            return None
        data: dict[str, Any] = json.loads(rows[0]["payload"])
        return data

    def save(
        self,
        endpoint: str,
        endpoint_hash: str,
        intents: list[TestIntent],
        tests: list[ValidatedTest],
    ) -> None:
        payload = {
            "intents": [i.model_dump() for i in intents],
            "tests": [t.model_dump() for t in tests],
        }
        self.store.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, 'plan', ?, strftime('%s','now'))",
            (self.api_key, endpoint, endpoint_hash, json.dumps(payload)),
        )


def batches(
    intents: list[TestIntent], difficulty: dict[str, float], size: int, max_difficulty: float = 0.45
) -> tuple[list[list[TestIntent]], list[TestIntent]]:
    """Group easy intents (ordered by endpoint, so a batch shares context) into batches of
    ``size``; harder intents go one by one through the cascade."""
    easy = sorted(
        (i for i in intents if difficulty.get(i.id, 1.0) <= max_difficulty),
        key=lambda i: (i.endpoint, i.id),
    )
    singles = [i for i in intents if difficulty.get(i.id, 1.0) > max_difficulty]
    out: list[list[TestIntent]] = []
    for k in range(0, len(easy), size):
        chunk = easy[k : k + size]
        if len(chunk) == 1:
            singles.extend(chunk)
        else:
            out.append(chunk)
    return out, singles
