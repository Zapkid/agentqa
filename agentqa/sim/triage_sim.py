"""Simulated triage, judge and injection classifier (rule-based reasoning + declared noise)."""

from __future__ import annotations

import re
from typing import Any

from agentqa.llm.simulated import SimRequest, responder

TEST_ERRORS = (
    "TypeError",
    "KeyError",
    "AttributeError",
    "NameError",
    "StopIteration",
    "IndexError",
    "JSONDecodeError",
    "SyntaxError",
)
ENV_ERRORS = ("ConnectError", "ConnectionRefused", "ReadTimeout", "database busy")


@responder("triage")
def triage(req: SimRequest) -> dict[str, Any]:
    p = req.payload()
    msg = p.get("failure_message", "")
    documented = {int(s) for s in p.get("documented_statuses", []) if str(s).isdigit()}
    intent = p.get("intent", {})
    refs = list(p.get("evidence", {}))
    risk = int(intent.get("risk", 3))
    severity = {5: "critical", 4: "high", 3: "medium"}.get(risk, "low")
    confidence = 0.85
    m = re.search(r"assert (\d{3}) == (\d{3})", msg)
    if any(e in msg for e in ENV_ERRORS):
        cls, why = "env_issue", "the environment failed (connection or capacity), not the API logic"
    elif any(e in msg for e in TEST_ERRORS):
        cls, why = "test_bug", "the test raised a Python error before reaching its assertion"
    elif m and documented and int(m.group(2)) not in documented:
        cls, why = (
            "test_bug",
            f"the test expects status {m.group(2)}, which the spec does not document",
        )
    else:
        cls = "product_bug"
        why = (
            f"the API returned {m.group(1)} where the requirement expects {m.group(2)}"
            if m
            else "the response violates the documented expectation: " + msg.splitlines()[-1][:160]
            if msg
            else ""
        )
        confidence = 0.8 if m else 0.65
    if req.mistake(0.5):
        cls = {"product_bug": "test_bug", "test_bug": "product_bug"}.get(cls, cls)
        confidence = 0.55
    cite = [r for r in refs if r.startswith(("exchange:", "spec:", "doc:"))][:3]
    if req.mistake(0.2):
        cite = ["exchange:made-up:9"]  # cites evidence it was not given -> downgraded
    return {
        "classification": cls,
        "severity": severity if cls == "product_bug" else "low",
        "root_cause_hypothesis": why,
        "evidence_refs": cite,
        "confidence": confidence,
        "title": f"{intent.get('title', 'Failure')}"
        if cls == "product_bug"
        else f"{intent.get('title', 'Test')} (test defect)",
    }


@responder("judge")
def judge(req: SimRequest) -> dict[str, Any]:
    p = req.payload()
    findings = p.get("findings", [])
    unsupported = [f["id"] for f in findings if not f.get("evidence")]
    weak = [
        f["id"]
        for f in findings
        if f.get("classification") == "product_bug" and f.get("confidence", 1) < 0.6
    ]
    accuracy = 5 - min(4, len(unsupported) + len(weak))
    actionability = 5 if all(f.get("repro_curl", "").startswith("curl") for f in findings) else 3
    clarity = 4
    if req.mistake(0.3):
        accuracy = max(1, accuracy - 1)
    return {
        "accuracy": accuracy,
        "actionability": actionability,
        "clarity": clarity,
        "unsupported_finding_ids": unsupported,
        "comments": "simulated judge",
    }


@responder("injection_classifier")
def classify(req: SimRequest) -> dict[str, Any]:
    text = req.user.lower()
    signals = (
        "as an ai",
        "assistant",
        "instructions",
        "you must now",
        "tool",
        "jailbreak",
        "system prompt",
    )
    hit = sum(s in text for s in signals)
    is_inj = hit >= 2
    if req.mistake(0.3):
        is_inj = not is_inj
    return {
        "is_injection": is_inj,
        "confidence": 0.7,
        "reason": f"{hit} signals addressed to an AI system",
    }
