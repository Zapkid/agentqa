"""Simulated planner: plans an intent when a retrieved doc chunk states the rule."""

from __future__ import annotations

from typing import Any

from agentqa.llm.simulated import SimRequest, responder
from agentqa.sim.common import documents
from agentqa.sim.library import RULES


@responder("planner")
def plan(req: SimRequest) -> dict[str, Any]:
    payload = req.payload()
    endpoint_id = payload.get("endpoint_id", "")
    spec_ref = payload.get("spec_chunk", {}).get("id", f"spec:{endpoint_id}")
    docs = documents(req.user)
    allowed = set(payload.get("categories", []))
    intents = []
    for rule in RULES:
        if f"{rule.method} {rule.path}" != endpoint_id or rule.category not in allowed:
            continue
        support = [
            cid for cid, text in docs.items() if all(k.lower() in text for k in rule.keywords)
        ]
        if not support:
            continue  # the rule was not in the retrieved context: a real model could not plan it either
        refs = [spec_ref, support[0]]
        if req.mistake(0.4):
            if req.rng.random() < 0.5:
                continue  # missed the intent
            refs = [f"doc:{rule.key}-notes#section"]  # cites a chunk it was never given
        intents.append(
            {
                "category": rule.category,
                "title": rule.title,
                "risk": rule.risk,
                "rationale": f"Requirement in {support[0]} states this rule; violating it affects {rule.category}.",
                "preconditions": list(rule.preconditions),
                "expected_behavior": rule.expected,
                "source_refs": refs,
            }
        )
    return {"intents": intents[: int(payload.get("max_intents", 6))]}
