"""Simulated generator: looks up the template for an intent, uses tools on the first turn, and
injects declared mistakes (hallucinated path, unknown field, undocumented status, wrong logic)."""

from __future__ import annotations

import json
from typing import Any

from agentqa.llm.simulated import SimRequest, responder
from agentqa.llm.types import ToolCall
from agentqa.sim.library import RULES_BY_TITLE, Rule

MISTAKES = ("hallucinated_path", "unknown_field", "undocumented_status", "wrong_logic")


def _render(rule: Rule, name: str) -> dict[str, Any]:
    return {
        "test_name": name,
        "code": rule.code.replace("TEST_NAME", name).strip() + "\n",
        "requests_made": [
            {"method": m, "path": p, "fields": list(f), "expected_status": list(s)}
            for m, p, f, s in rule.requests
        ],
        "confidence": 0.85,
    }


def _generic(intent: dict[str, Any], name: str) -> dict[str, Any]:
    method, path = intent["endpoint"].split(" ", 1)
    code = f"""
def {name}(client, auth):
    r = client.request({method!r}, {path!r}, headers=auth["admin"])
    assert r.status_code < 500
"""
    return {
        "test_name": name,
        "code": code.strip() + "\n",
        "requests_made": [{"method": method, "path": path, "fields": [], "expected_status": []}],
        "confidence": 0.35,
    }


def _inject(out: dict[str, Any], kind: str) -> dict[str, Any]:
    code: str = out["code"]
    decls: list[dict[str, Any]] = out["requests_made"]
    if kind == "hallucinated_path":
        code = code.replace('"/orders"', '"/v1/orders"', 1).replace(
            '"/products"', '"/v1/products"', 1
        )
        decls = [{**d, "path": "/v1" + d["path"]} if i == 0 else d for i, d in enumerate(decls)]
    elif kind == "unknown_field":
        code = code.replace('{"items":', '{"line_items":', 1).replace(
            '{"price":', '{"unit_price":', 1
        )
        decls = [
            {
                **d,
                "fields": [
                    "line_items" if f == "items" else "unit_price" if f == "price" else f
                    for f in d["fields"]
                ],
            }
            for d in decls
        ]
    elif kind == "undocumented_status":
        decls = [
            {**d, "expected_status": [418]} if i == len(decls) - 1 else d
            for i, d in enumerate(decls)
        ]
    elif kind == "wrong_logic":
        for good, bad in (
            ("== 404", "== 403"),
            ("== 409", "== 200"),
            ("== 422", "== 400"),
            ("== 401", "== 403"),
            ("ROUND_HALF_UP", "decimal.ROUND_HALF_EVEN"),
            ("timedelta(hours=2)", "timedelta(hours=0)"),
            ("== 201", "== 200"),
        ):
            if good in code:
                code = code.replace(good, bad, 1)
                break
        out = {**out, "confidence": max(0.2, out["confidence"] - 0.1)}
    return {**out, "code": code, "requests_made": decls}


def _write(req: SimRequest, intent: dict[str, Any], name: str, feedback: bool) -> dict[str, Any]:
    rule = RULES_BY_TITLE.get(intent.get("title", "").lower())
    out = _render(rule, name) if rule else _generic(intent, name)
    difficulty = rule.difficulty if rule else 0.9
    if (
        "Lessons from earlier runs" in req.user
        and intent.get("endpoint", "~") in req.user.split("Lessons from earlier runs", 1)[1]
    ):
        difficulty *= 0.5  # simulated learning: lessons about this endpoint halve the error rate
    if req.mistake(difficulty):
        kinds = (
            MISTAKES if not feedback else ("wrong_logic",)
        )  # feedback fixes declared/grounding errors
        out = _inject(out, kinds[req.rng.randrange(len(kinds))])
    return out


@responder("generator")
def generate(req: SimRequest) -> dict[str, Any] | list[ToolCall]:
    payload = req.payload()
    intent = payload.get("intent", {})
    if not req.tool_results():
        return [
            ToolCall(
                id="t1",
                name="get_endpoint_schema",
                arguments={"endpoint_id": intent.get("endpoint", "")},
            ),
            ToolCall(id="t2", name="list_fixtures", arguments={}),
        ]
    feedback = "rejected by automated checks" in req.user
    return _write(req, intent, payload.get("test_name_hint", "test_generated"), feedback)


@responder("generator_batch")
def generate_batch(req: SimRequest) -> dict[str, Any]:
    payload = req.payload()
    tests = []
    for item in payload.get("intents", []):
        intent = item["intent"]
        out = _write(req, intent, item.get("test_name_hint", "test_generated"), False)
        tests.append({"intent_id": intent["id"], **out})
    return {"tests": tests}


def dumps(obj: Any) -> str:
    return json.dumps(obj)
