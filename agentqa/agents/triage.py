"""F6 Triage agent: cluster first, then one LLM call per cluster with a compact evidence bundle.

Deterministic first:
- blocked by the sandbox -> test_bug (rule), listed with the guardrail event
- collection/runtime errors in the test itself -> test_bug (rule)
- mixed outcomes across reruns -> flaky (rule)
Remaining failures are clustered by (endpoint, symptom). Each cluster gets one call at T1; low
confidence or critical severity escalates the same cluster to T2. A verdict whose evidence refs
are not in the bundle is downgraded to ``needs_review`` (no unsupported claims).
"""

from __future__ import annotations

import json
import re
import shlex
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from agentqa.guards.injection import delimit
from agentqa.llm.normalize import normalize
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient
from agentqa.llm.types import LLMError
from agentqa.models import (
    Evidence,
    Finding,
    HttpExchange,
    RunResult,
    SpecBundle,
    TestIntent,
    TestResult,
    TriageVerdict,
)
from agentqa.obs import metrics, tracing

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
TEST_ERRORS = (
    "TypeError",
    "KeyError",
    "AttributeError",
    "NameError",
    "StopIteration",
    "IndexError",
    "JSONDecodeError",
    "collection error",
    "SyntaxError",
    "fixture",
)
ENV_ERRORS = ("ConnectError", "ConnectionRefused", "ReadTimeout", "database busy", "TIMEOUT")


@dataclass
class Cluster:
    key: str
    endpoint: str
    symptom: str
    results: list[TestResult] = field(default_factory=list)
    intents: list[TestIntent] = field(default_factory=list)


def symptom(result: TestResult) -> str:
    msg = result.message.splitlines()[-1] if result.message else ""
    msg = UUID_RE.sub("<id>", msg)
    msg = re.sub(r"/tmp/\S+", "", msg)
    msg = re.sub(r"\d+\.\d+", "<n>", msg)
    status = next((e.status for e in reversed(result.exchanges) if e.status is not None), None)
    exc = next(
        (t for t in (*TEST_ERRORS, *ENV_ERRORS, "AssertionError") if t in result.message), "other"
    )
    return f"{exc}|status={status}|{msg[:80]}"


def cluster_failures(
    failures: list[tuple[TestResult, TestIntent]], enabled: bool = True
) -> list[Cluster]:
    """Cluster by (endpoint, symptom). Tier-0 checks of the same kind failing with the same symptom
    on several endpoints share one root cause (e.g. one id parser used everywhere) and are
    merged into a single cluster across endpoints."""
    clusters: OrderedDict[str, Cluster] = OrderedDict()
    for result, intent in failures:
        sym = symptom(result)
        if not enabled:
            key = f"{intent.endpoint}::{result.test_name}"
        elif intent.origin == "t0":
            key = f"t0:{intent.id.split('-')[1]}::{sym}"
        else:
            key = f"{intent.endpoint}::{sym}"
        c = clusters.setdefault(key, Cluster(key=key, endpoint=intent.endpoint, symptom=sym))
        c.results.append(result)
        c.intents.append(intent)
    return list(clusters.values())


def curl_for(ex: HttpExchange | None) -> str:
    if ex is None:
        return "# no HTTP exchange recorded"
    parts = ["curl", "-sS", "-X", ex.method, shlex.quote(ex.url)]
    for k, v in ex.request_headers.items():
        if k.lower() in (
            "host",
            "content-length",
            "accept-encoding",
            "connection",
            "user-agent",
            "accept",
        ):
            continue
        if v == "[REDACTED]" and k.lower() == "authorization":
            v = "Bearer $TOKEN"
        elif v == "[REDACTED]":
            v = f"${k.upper().replace('-', '_')}"
        parts += ["-H", shlex.quote(f"{k}: {v}")]
    if ex.request_body:
        parts += ["--data", shlex.quote(ex.request_body)]
    return " ".join(parts)


class Triage:
    def __init__(
        self, bundle: SpecBundle, *, escalate_confidence: float = 0.6, cluster_first: bool = True
    ) -> None:
        self.bundle = bundle
        self.chunks = {c.id: c for c in bundle.chunks}
        self.prompt = load_prompt("triage")
        self.escalate_confidence = escalate_confidence
        self.cluster_first = cluster_first
        self.stats: dict[str, int] = {
            "clusters": 0,
            "llm_calls": 0,
            "escalations": 0,
            "downgraded": 0,
            "rule_classified": 0,
        }

    # ------------------------------------------------------------------ evidence

    def evidence_bundle(
        self, c: Cluster, run: RunResult
    ) -> tuple[dict[str, Any], dict[str, Evidence]]:
        rep, intent = c.results[0], c.intents[0]
        evidence: dict[str, Evidence] = {}
        for i, ex in enumerate(rep.exchanges[-3:]):
            ref = f"exchange:{rep.test_name}:{i}"
            evidence[ref] = Evidence(
                kind="request",
                ref=ref,
                excerpt=(
                    f"{ex.method} {ex.path} -> {ex.status}\nrequest: {(ex.request_body or '')[:300]}\n"
                    f"response: {(ex.response_body or '')[:500]}"
                ),
            )
        spec_ref = f"spec:{intent.endpoint}"
        if spec_ref in self.chunks:
            evidence[spec_ref] = Evidence(
                kind="spec_clause", ref=spec_ref, excerpt=self.chunks[spec_ref].text[:1200]
            )
        for ref in intent.source_refs:
            ch = self.chunks.get(ref)
            if ch is not None and ch.kind == "doc" and not ch.quarantined:
                evidence[ref] = Evidence(kind="doc_clause", ref=ref, excerpt=ch.text[:900])
        paths = {ex.path for ex in rep.exchanges}
        log_lines = [ln for ln in run.server_log if any(p and p in ln for p in paths)][-4:]
        errors = [ln for ln in run.server_log if '"level": "error"' in ln][-2:]
        for i, ln in enumerate([*log_lines, *errors]):
            evidence[f"log:{i}"] = Evidence(kind="log_line", ref=f"log:{i}", excerpt=ln[:400])
        evidence[f"rerun:{rep.test_name}"] = Evidence(
            kind="rerun",
            ref=f"rerun:{rep.test_name}",
            excerpt=f"outcome={rep.outcome}; reruns={rep.reruns}",
        )
        try:
            documented = sorted(self.bundle.endpoint(intent.endpoint).status_codes)
        except KeyError:
            documented = []
        payload = {
            "cluster": {
                "endpoint": c.endpoint,
                "symptom": c.symptom,
                "size": len(c.results),
                "tests": [r.test_name for r in c.results][:8],
            },
            "intent": {
                "title": intent.title,
                "category": intent.category,
                "risk": intent.risk,
                "expected_behavior": intent.expected_behavior,
                "origin": intent.origin,
            },
            "failure_message": re.sub(r"/\S*/suite/", "", rep.message)[-900:],
            "documented_statuses": documented,
            "evidence": {
                ref: (e.excerpt if e.kind != "doc_clause" else delimit(ref, e.excerpt))
                for ref, e in evidence.items()
            },
        }
        return payload, evidence

    # ------------------------------------------------------------------ classification

    def _rule(self, c: Cluster) -> tuple[str, str] | None:
        rep = c.results[0]
        if rep.outcome == "blocked":
            return "test_bug", "the test attempted an action the sandbox blocks"
        if rep.flaky:
            return "flaky", f"outcome changed across reruns: {rep.outcome} then {rep.reruns}"
        if rep.outcome == "error":
            return "test_bug", "the test could not be collected or set up"
        return None

    def _call(self, client: LLMClient, payload: dict[str, Any], task_id: str) -> TriageVerdict:
        self.stats["llm_calls"] += 1
        result = client.complete(
            self.prompt.render(payload=normalize(json.dumps(payload, indent=1))),
            response_schema=TriageVerdict,
            max_tokens=900,
            metadata=self.prompt.metadata("triage", task_id=task_id),
        )
        return cast(TriageVerdict, result.parsed)

    def triage(
        self,
        run: RunResult,
        intents: dict[str, TestIntent],
        client_for_tier: Callable[[str], LLMClient],
        on_delegation: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[Finding]:
        failures = [
            (r, intents[r.intent_id])
            for r in run.results
            if r.outcome in ("failed", "error", "blocked") and r.intent_id in intents
        ]
        clusters = cluster_failures(failures, enabled=self.cluster_first)
        self.stats["clusters"] = len(clusters)
        findings: list[Finding] = []
        with tracing.span(
            "triage",
            "stage",
            **{"agentqa.failures": len(failures), "agentqa.clusters": len(clusters)},
        ):
            for n, c in enumerate(clusters, start=1):
                payload, evidence = self.evidence_bundle(c, run)
                rule = self._rule(c)
                rep_ex = next(
                    (e for e in reversed(c.results[0].exchanges) if e.status is not None), None
                )
                endpoints = list(dict.fromkeys(i.endpoint for i in c.intents))
                common: dict[str, Any] = dict(
                    endpoint=c.endpoint,
                    endpoints=endpoints,
                    category=c.intents[0].category,
                    repro_curl=curl_for(rep_ex),
                    test_names=[r.test_name for r in c.results],
                    intent_ids=[i.id for i in c.intents],
                )
                if rule is not None:
                    self.stats["rule_classified"] += 1
                    findings.append(
                        Finding(
                            id=f"F-{n:03d}",
                            title=f"{c.intents[0].title} ({rule[0].replace('_', ' ')})",
                            classification=rule[0],  # type: ignore[arg-type]
                            severity="low",
                            root_cause_hypothesis=rule[1],
                            evidence=list(evidence.values())[:4],
                            confidence=1.0,
                            triaged_by="rule",
                            **common,
                        )
                    )
                    continue
                task_id = f"triage-{n}"
                tier = "T1"
                try:
                    verdict = self._call(client_for_tier("T1"), payload, task_id)
                    if (
                        verdict.confidence < self.escalate_confidence
                        or verdict.severity == "critical"
                    ):
                        reason = (
                            "low_confidence"
                            if verdict.confidence < self.escalate_confidence
                            else "critical_severity"
                        )
                        self.stats["escalations"] += 1
                        metrics.inc(
                            "agentqa_escalations_total", from_tier="T1", to_tier="T2", reason=reason
                        )
                        tracing.event(
                            "delegation.escalate",
                            task_id=task_id,
                            from_tier="T1",
                            to_tier="T2",
                            reason=reason,
                        )
                        verdict = self._call(client_for_tier("T2"), payload, task_id)
                        tier = "T2"
                except LLMError as exc:
                    tracing.event("triage.failed", cluster=c.key, error=str(exc)[:200])
                    findings.append(
                        Finding(
                            id=f"F-{n:03d}",
                            title=f"{c.intents[0].title} (not triaged)",
                            classification="needs_review",
                            severity="medium",
                            root_cause_hypothesis=f"triage failed: {type(exc).__name__}",
                            evidence=list(evidence.values())[:4],
                            confidence=0.0,
                            triaged_by="none",
                            **common,
                        )
                    )
                    continue
                cited = [evidence[r] for r in verdict.evidence_refs if r in evidence]
                classification: str = verdict.classification
                if not cited:
                    classification = "needs_review"
                    self.stats["downgraded"] += 1
                    metrics.inc(
                        "agentqa_guardrail_events_total", guardrail="evidence", action="downgrade"
                    )
                    tracing.event(
                        "triage.downgraded",
                        cluster=c.key,
                        reason="no valid evidence refs",
                        refs=verdict.evidence_refs,
                    )
                if on_delegation:
                    on_delegation(
                        {
                            "task_id": task_id,
                            "task_type": "triage",
                            "tier": tier,
                            "outcome": classification,
                            "confidence": verdict.confidence,
                        }
                    )
                findings.append(
                    Finding(
                        id=f"F-{n:03d}",
                        title=verdict.title,
                        classification=classification,  # type: ignore[arg-type]
                        severity=verdict.severity,
                        root_cause_hypothesis=verdict.root_cause_hypothesis,
                        evidence=cited or list(evidence.values())[:3],
                        confidence=verdict.confidence,
                        triaged_by=tier,
                        **common,
                    )
                )
            for f in findings:
                labels = {"class": f.classification, "severity": f.severity}
                metrics.inc("agentqa_findings_total", 1, **labels)
        return findings
