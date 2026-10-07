"""Simulated workload modeller and perf triage (rule-based reasoning + declared noise)."""

from __future__ import annotations

import re
from typing import Any

from agentqa.llm.simulated import SimRequest, responder
from agentqa.sim.common import documents


def _num(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1).replace(",", "")) if m else None


@responder("workload")
def workload(req: SimRequest) -> dict[str, Any]:
    p = req.payload()
    endpoints = {e["id"] for e in p.get("endpoints", [])}
    docs = documents(req.user)
    slo_doc = next((cid for cid, t in docs.items() if "p95" in t), None)
    mix_doc = next(
        (cid for cid, t in docs.items() if "traffic mix" in t or "% order list" in t), None
    )
    text = docs.get(slo_doc or "", "") + " " + docs.get(mix_doc or "", "")
    scen: list[dict[str, Any]] = []

    def add(
        name: str,
        method: str,
        path: str,
        weight: int,
        expected: list[int],
        params: dict[str, Any] | None = None,
        body: str = "none",
    ) -> None:
        if f"{method} {path}" in endpoints:
            scen.append(
                {
                    "name": name,
                    "method": method,
                    "path": path,
                    "params": params or {},
                    "body": body,
                    "weight": weight,
                    "expected_status": expected,
                }
            )

    list_w = int(_num(r"(\d+)% order list", text) or 50)
    add("list_orders", "GET", "/orders", max(1, list_w // 2), [200])
    add(
        "list_orders_filtered",
        "GET",
        "/orders",
        max(1, list_w // 4),
        [200],
        {"status": "paid", "created_from": "2025-11-01T00:00:00Z"},
    )
    add("list_orders_page100", "GET", "/orders", max(1, list_w // 4), [200], {"page_size": 100})
    add("list_orders_oversized", "GET", "/orders", 2, [200, 422], {"page_size": 2000})
    add(
        "get_order",
        "GET",
        "/orders/{order_id}",
        int(_num(r"(\d+)% single order", text) or 15),
        [200, 404],
    )
    add("list_products", "GET", "/products", int(_num(r"(\d+)% product", text) or 10), [200])
    add(
        "create_order",
        "POST",
        "/orders",
        int(_num(r"(\d+)% order creation", text) or 10),
        [201],
        body="order",
    )
    add(
        "payment_webhook",
        "POST",
        "/webhooks/payment",
        int(_num(r"(\d+)% webhooks", text) or 5),
        [200, 404, 409],
        body="webhook",
    )
    if req.mistake(0.4):
        scen[0] = {
            **scen[0],
            "path": "/orders/list",
        }  # invented path: validation rejects, escalation fixes
    slos = []
    if slo_doc:
        slos.append(
            {
                "p95_ms": _num(r"p95 under (\d+) ?ms", text),
                "p99_ms": _num(r"p99 under (\d+) ?ms", text),
                "error_rate": (_num(r"error rate under (\d+(?:\.\d+)?)%", text) or 1) / 100,
                "min_rps": None,
                "scope": "list_orders",
            }
        )
        rps = _num(r"at least (\d+) requests per second", text)
        if rps:
            slos.append(
                {
                    "p95_ms": None,
                    "p99_ms": None,
                    "error_rate": None,
                    "min_rps": rps,
                    "scope": "payment_webhook",
                }
            )
    users = _num(r"(\d+) concurrent users", text)
    rows = _num(r"up to ([\d,]+) orders", text)
    return {
        "scenarios": scen,
        "think_time_s": 0.0,
        "data_setup_orders": int(rows or 0),
        "ramp_profile": "stepped ramp to and beyond the documented peak",
        "slos": slos,
        "slo_source": [slo_doc] if slo_doc else [],
        "peak_users": int(users or 20),
    }


@responder("perf_triage")
def perf_triage(req: SimRequest) -> dict[str, Any]:
    """Reads the same numbers a person would and applies the textbook signatures, with noise."""
    from agentqa.perf.analysis import rule_diagnosis
    from agentqa.perf.triage import FIXES

    bundle = req.payload()
    try:
        cls, keys = rule_diagnosis(bundle)
    except (KeyError, TypeError):
        cls, keys = "none", []
    confidence = 0.85 if cls != "none" else 0.6
    if req.mistake(0.5):
        wrong = [
            "n_plus_one",
            "missing_index",
            "blocking_handler",
            "pool_exhaustion",
            "memory_leak",
            "unbounded_payload",
        ]
        cls = wrong[req.rng.randrange(len(wrong))]
        confidence = 0.5
    return {
        "bottleneck": cls,
        "confidence": confidence,
        "evidence_keys": keys,
        "explanation": f"evidence {keys} matches the {cls} signature"
        if keys
        else "no regression signature",
        "fix": FIXES.get(cls, "no change needed"),
    }
