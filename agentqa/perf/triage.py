"""Perf triage: one small-model call per compact evidence bundle (never raw logs), escalating to
the strong tier on low confidence or when the model's class has no supporting evidence key."""

from __future__ import annotations

import json
from typing import Any, cast

from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient
from agentqa.llm.types import LLMError
from agentqa.obs import metrics, tracing
from agentqa.perf.models import PerfVerdict

FIXES = {
    "n_plus_one": "Load related rows in one query (JOIN or WHERE id IN (...)) instead of one query per row.",
    "missing_index": "Add indexes on the filtered/sorted columns (e.g. orders(status, created_at)) and check EXPLAIN.",
    "unbounded_payload": "Enforce the documented page_size maximum and paginate large responses.",
    "blocking_handler": "Move blocking work out of the event loop (async I/O, a worker thread, or a queue).",
    "pool_exhaustion": "Size the DB pool for the concurrency target and use a longer acquire timeout with backpressure.",
    "memory_leak": "Bound or remove the per-request cache (LRU with a size cap) and watch RSS in a soak test.",
}


def diagnose(
    bundle: dict[str, Any], client_for_tier: dict[str, LLMClient], task_id: str
) -> tuple[PerfVerdict, list[str]]:
    prompt = load_prompt("perf_triage")
    path: list[str] = []
    last: PerfVerdict | None = None
    for tier in ("T1", "T2"):
        try:
            with tracing.span(
                "agent perf_triage",
                "agent",
                **{"agentqa.agent": "perf_triage", "agentqa.tier": tier},
            ):
                res = client_for_tier[tier].complete(
                    prompt.render(payload=json.dumps(bundle, indent=1)),
                    response_schema=PerfVerdict,
                    max_tokens=700,
                    metadata=prompt.metadata("perf_triage", task_id=task_id),
                )
            v = cast(PerfVerdict, res.parsed)
        except LLMError as exc:
            path.append(f"{tier}:error")
            tracing.event("perf_triage.failed", tier=tier, error=str(exc)[:200])
            continue
        unsupported = v.bottleneck != "none" and not any(k in bundle for k in v.evidence_keys)
        if v.confidence >= 0.6 and not unsupported:
            path.append(f"{tier}:ok")
            return v, path
        reason = "unsupported" if unsupported else "low_confidence"
        path.append(f"{tier}:{reason}")
        last = v
        if tier == "T1":
            metrics.inc("agentqa_escalations_total", from_tier="T1", to_tier="T2", reason=reason)
            tracing.event(
                "delegation.escalate", task_id=task_id, from_tier="T1", to_tier="T2", reason=reason
            )
    if last is None:
        last = PerfVerdict(
            bottleneck="none", confidence=0.0, explanation="perf triage failed", fix=""
        )
    return last, path
