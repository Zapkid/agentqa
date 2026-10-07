"""Workload modelling agent: one small-model call per run, escalating only if the WorkloadSpec
fails validation (schema, or paths/params that are not in the spec)."""

from __future__ import annotations

import json
from typing import cast

from agentqa.guards.grounding import check_decl
from agentqa.guards.injection import delimit
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient
from agentqa.llm.types import LLMError
from agentqa.models import GuardViolation, RequestDecl, SpecBundle
from agentqa.obs import metrics, tracing
from agentqa.perf.models import WorkloadSpec


def validate(spec: WorkloadSpec, bundle: SpecBundle) -> list[GuardViolation]:
    out: list[GuardViolation] = []
    for s in spec.scenarios:
        out += check_decl(
            RequestDecl(
                method=s.method,
                path=s.path,
                fields=list(s.params),
                expected_status=s.expected_status,
            ),
            bundle.endpoints,
        )
    known_chunks = {c.id for c in bundle.chunks}
    out += [
        GuardViolation(kind="unknown_source", detail=f"slo_source {c!r} not provided")
        for c in spec.slo_source
        if c not in known_chunks
    ]
    return out


def model_workload(
    bundle: SpecBundle, store: VectorStore, client_for_tier: dict[str, LLMClient]
) -> tuple[WorkloadSpec, list[str]]:
    """Returns (spec, path) where path records tiers tried, e.g. ['T1:unknown_path', 'T2:ok']."""
    prompt = load_prompt("workload")
    hits = store.hybrid(
        bundle.spec_id,
        [
            "performance service levels latency p95 concurrent users throughput",
            "expected traffic mix at peak",
        ],
        k=4,
        kind="doc",
    )
    chunks = {c.id: c for c in bundle.chunks}
    docs = "\n".join(delimit(cid, chunks[cid].text) for cid, _ in hits if cid in chunks)
    payload = {
        "endpoints": [
            {
                "id": e.id,
                "params": [p.name for p in e.params if p.location == "query"],
                "requires_auth": e.requires_auth,
            }
            for e in bundle.endpoints
        ],
        "document_ids": [cid for cid, _ in hits],
        "body_kinds": {
            "order": "creates a small order",
            "webhook": "a signed payment webhook event",
            "example": "an example body from the spec",
            "none": "no body",
        },
    }
    path: list[str] = []
    feedback = ""
    for tier in ("T1", "T2"):
        messages = prompt.render(payload=json.dumps(payload, indent=1) + feedback, documents=docs)
        try:
            with tracing.span(
                "agent workload", "agent", **{"agentqa.agent": "workload", "agentqa.tier": tier}
            ):
                res = client_for_tier[tier].complete(
                    messages,
                    response_schema=WorkloadSpec,
                    max_tokens=1500,
                    metadata=prompt.metadata("workload", task_id="workload"),
                )
            spec = cast(WorkloadSpec, res.parsed)
            violations = validate(spec, bundle)
        except LLMError as exc:
            violations = [GuardViolation(kind="generation_error", detail=str(exc)[:200])]
        if not violations:
            path.append(f"{tier}:ok")
            return spec, path
        path.append(f"{tier}:{violations[0].kind}")
        feedback = "\n\nYour previous workload was rejected:\n" + "\n".join(
            f"- {v.kind}: {v.detail}" for v in violations
        )
        if tier == "T1":
            metrics.inc(
                "agentqa_escalations_total", from_tier="T1", to_tier="T2", reason=violations[0].kind
            )
    raise LLMError(f"no valid workload after escalation: {path}")
