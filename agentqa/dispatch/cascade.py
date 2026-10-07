"""Verification-gated cascade and the delegation ledger.

Cheap verifiers, in order: static checks -> grounding guard -> self-reported confidence ->
execution on a reference build (when one is configured: the test must pass against a build
believed correct). On failure: repair once at the same tier with the failure feedback, then
escalate to the next tier with the full history attached. Never escalate on vibes: every
escalation names the verifier that failed. A task that still fails is quarantined and reported
as uncovered with the reason.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentqa.executor.runner import Executor
from agentqa.guards import grounding
from agentqa.guards.static_checks import check_code
from agentqa.llm.normalize import normalize
from agentqa.llm.types import BudgetExceeded, KillSwitchEngaged, LLMError
from agentqa.models import Endpoint, GeneratedTest, GuardViolation, TestIntent, ValidatedTest
from agentqa.obs import metrics, tracing
from agentqa.store import Store

GROUNDING_KINDS = {
    "unknown_path",
    "method_not_allowed",
    "unknown_field",
    "undocumented_status",
    "undeclared_request",
}
MIN_CONFIDENCE = 0.3


class ReferenceRunner:
    """Runs one generated test against a reference (believed-correct) build."""

    def __init__(self, executor: Executor, base_url: str, work_dir: Path) -> None:
        self.executor = executor
        self.base_url = base_url
        self.work_dir = work_dir
        self.runs = 0

    def check(self, vt: ValidatedTest) -> GuardViolation | None:
        self.runs += 1
        out = self.work_dir / f"{vt.test.test_name}-{self.runs}"
        result = self.executor.run([vt], self.base_url, out).results[0]
        if result.outcome in ("passed", "skipped"):
            return None
        return GuardViolation(
            kind="fails_on_reference",
            detail=f"{result.outcome} on the reference build: {result.message[-300:]}",
        )


@dataclass
class Verifier:
    endpoints: list[Endpoint]
    reference: ReferenceRunner | None = None
    min_confidence: float = MIN_CONFIDENCE

    def verify(self, intent: TestIntent, test: GeneratedTest, tier: str) -> list[GuardViolation]:
        with tracing.span(f"verify {test.test_name}", "guard", **{"agentqa.tier": tier}) as sp:
            violations = check_code(test.code, test.test_name) + grounding.check_test(
                test, self.endpoints
            )
            if not violations and test.confidence < self.min_confidence:
                violations.append(
                    GuardViolation(
                        kind="low_confidence", detail=f"self-reported {test.confidence:.2f}"
                    )
                )
            if not violations and self.reference is not None:
                failed = self.reference.check(ValidatedTest(intent=intent, test=test, tier=tier))
                if failed:
                    violations.append(failed)
            sp.set_attribute(
                "agentqa.violations", [f"{v.kind}: {v.detail[:120]}" for v in violations]
            )
            return violations


def feedback_text(history: list[tuple[str, list[GuardViolation]]]) -> str:
    lines = []
    for label, violations in history:
        lines.append(f"Attempt {label} was rejected:")
        lines.extend(f"- {v.kind}: {v.detail}" for v in violations)
    return normalize("\n".join(lines))


@dataclass
class CascadeOutcome:
    validated: ValidatedTest | None
    path: list[str] = field(default_factory=list)
    pre_repair: list[GuardViolation] = field(default_factory=list)
    uncovered_reason: str | None = None
    tokens: int = 0

    @property
    def hallucinated(self) -> bool:
        return any(v.kind in GROUNDING_KINDS for v in self.pre_repair)


Generate = Callable[
    [TestIntent, str, str | None], GeneratedTest
]  # (intent, tier, feedback) -> test


def run_cascade(
    intent: TestIntent,
    start_tier: str,
    can_escalate: bool,
    generate: Generate,
    verifier: Verifier,
    tokens_used: Callable[[], int],
) -> CascadeOutcome:
    out = CascadeOutcome(validated=None)
    t0 = tokens_used()
    history: list[tuple[str, list[GuardViolation]]] = []
    tiers = [start_tier] + (["T2"] if can_escalate and start_tier == "T1" else [])
    for tier in tiers:
        for attempt in ("first", "repair"):
            label = f"{tier}:{attempt}"
            feedback = feedback_text(history) if history else None
            try:
                test = generate(intent, tier, feedback)
                violations = verifier.verify(intent, test, tier)
            except (KillSwitchEngaged, BudgetExceeded):
                raise
            except LLMError as exc:
                violations = [
                    GuardViolation(
                        kind="generation_error", detail=f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                ]
                test = None
            if not out.path:
                out.pre_repair = violations
                grounding.record(
                    "pre_repair", intent.id, [v for v in violations if v.kind in GROUNDING_KINDS]
                )
            out.path.append(f"{label}:{'ok' if not violations else violations[0].kind}")
            if not violations and test is not None:
                out.validated = ValidatedTest(
                    intent=intent,
                    test=test,
                    tier=tier,
                    violations_pre_repair=out.pre_repair,
                    repaired=len(out.path) > 1,
                    escalations=[p for p in out.path if "T2" in p],
                )
                out.tokens = tokens_used() - t0
                return out
            history.append((label, violations))
        if tier != tiers[-1]:
            reason = history[-1][1][0].kind if history[-1][1] else "unknown"
            metrics.inc("agentqa_escalations_total", from_tier=tier, to_tier="T2", reason=reason)
            tracing.event(
                "delegation.escalate",
                task_id=intent.id,
                from_tier=tier,
                to_tier="T2",
                reason=reason,
            )
    last = history[-1][1] if history else []
    out.uncovered_reason = "quarantined after verification failures: " + "; ".join(
        f"{v.kind}" for v in last[:3]
    )
    grounding.record("quarantine", intent.id, last)
    metrics.inc("agentqa_guardrail_events_total", guardrail="grounding", action="quarantine")
    out.tokens = tokens_used() - t0
    return out


class DelegationLedger:
    """Every delegation is a SQLite row, a span event and a metric."""

    def __init__(self, store: Store, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        task_id: str,
        task_type: str,
        features: dict[str, Any],
        start_tier: str,
        reason: str,
        est_tokens: int,
        actual_tokens: int,
        cost_usd: float,
        verifier_outcome: str,
        escalation_path: list[str],
        final_result: str,
    ) -> None:
        row = {
            "task_id": task_id,
            "task_type": task_type,
            "features": features,
            "start_tier": start_tier,
            "reason": reason,
            "est_tokens": est_tokens,
            "actual_tokens": actual_tokens,
            "cost_usd": cost_usd,
            "verifier_outcome": verifier_outcome,
            "escalation_path": escalation_path,
            "final_result": final_result,
        }
        self.rows.append(row)
        self.store.execute(
            "INSERT INTO delegations (run_id, task_id, task_type, features, start_tier, reason, est_tokens,"
            " actual_tokens, cost_usd, verifier_outcome, escalation_path, final_result, ts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.run_id,
                task_id,
                task_type,
                json.dumps(features),
                start_tier,
                reason,
                est_tokens,
                actual_tokens,
                cost_usd,
                verifier_outcome,
                json.dumps(escalation_path),
                final_result,
                time.time(),
            ),
        )
        final_tier = (
            escalation_path[-1].split(":")[0]
            if escalation_path and escalation_path[-1][:2] in ("T1", "T2")
            else start_tier
        )
        metrics.inc(
            "agentqa_tasks_total", tier=final_tier, task_type=task_type, outcome=final_result
        )
        tracing.event(
            "delegation",
            task_id=task_id,
            task_type=task_type,
            start_tier=start_tier,
            reason=reason,
            est_tokens=est_tokens,
            actual_tokens=actual_tokens,
            verifier=verifier_outcome,
            path=escalation_path,
            result=final_result,
        )

    def summary(self) -> dict[str, Any]:
        rows = self.rows
        escalated = [
            r
            for r in rows
            if any("T2" in p for p in r["escalation_path"]) and r["start_tier"] == "T1"
        ]
        by_tier: dict[str, int] = {}
        for r in rows:
            by_tier[r["start_tier"]] = by_tier.get(r["start_tier"], 0) + 1
        return {
            "tasks": len(rows),
            "start_tier": by_tier,
            "escalations": len(escalated),
            "escalation_rate": round(len(escalated) / len(rows), 3) if rows else 0.0,
            "quarantined": sum(r["final_result"] == "quarantined" for r in rows),
        }
