"""Report assembly at the end of a run."""

from __future__ import annotations

from datetime import UTC, datetime

from agentqa.agents.reporter import RunReport, Uncovered, judge_report
from agentqa.dispatch import savings
from agentqa.llm.types import BudgetExceeded, KillSwitchEngaged
from agentqa.memory import derive_lessons
from agentqa.models import Finding, RunResult, TestIntent, ValidatedTest
from agentqa.obs import metrics
from agentqa.orchestrator.graph import TaskGraph
from agentqa.orchestrator.state import PipelineState


class Reporting(PipelineState):
    """Collects the validated suite and renders the run report."""

    def validated_tests(self, graph: TaskGraph) -> list[ValidatedTest]:
        tests: list[ValidatedTest] = list(graph.result("t0", []) or [])
        tests += self.reused_tests
        for t in graph.of_kind("gen"):
            if t.status == "done" and t.result is not None:
                tests.append(t.result)
            elif t.status == "failed":
                intent_id = t.id.split(":", 1)[1]
                self.uncovered.append(
                    Uncovered(
                        intent_id=intent_id,
                        endpoint="?",
                        category="?",
                        risk=0,
                        reason=f"generation task failed: {t.error}",
                    )
                )
        for t in graph.of_kind("batch"):
            if t.status == "done":
                tests += [v for v in t.result if v is not None]
        seen: set[str] = set()
        unique = []
        for vt in tests:
            if vt.test.test_name not in seen:
                seen.add(vt.test.test_name)
                unique.append(vt)
        return unique

    def finish(self, graph: TaskGraph, aborted: str | None, trace_id: str) -> RunReport:
        tests = self.validated_tests(graph)
        run: RunResult | None = graph.result("execute")
        findings: list[Finding] = graph.result("triage", []) or []
        intents: dict[str, TestIntent] = {vt.intent.id: vt.intent for vt in tests}
        d = graph.result("dispatch") or {}
        for i in d.get("planned", []):
            intents.setdefault(i.id, i)
        for u in list(self.uncovered):
            if u.endpoint == "?" and u.intent_id in intents:
                i = intents[u.intent_id]
                u.endpoint, u.category, u.risk = i.endpoint, i.category, i.risk
        covered = {vt.intent.id for vt in tests} | {u.intent_id for u in self.uncovered}
        if aborted:
            for i in intents.values():
                if i.id not in covered:
                    self._uncover(i, f"run aborted: {aborted}")
        if run is None and aborted:
            for vt in tests:
                self._uncover(vt.intent, f"not executed: run aborted: {aborted}")
        if self.bundle and not aborted:  # always save; reuse is what the mechanism gates
            inc = savings.Incremental(self.store, self.bundle.title)
            by_ep: dict[str, list[ValidatedTest]] = {}
            for vt in tests:
                if vt.tier != "T0":
                    by_ep.setdefault(vt.intent.endpoint, []).append(vt)
            for t in graph.of_kind("plan"):
                if t.status == "done":
                    ep_id = t.id.split(":", 1)[1]
                    key = (
                        f"{self.bundle.endpoint_hashes[ep_id]}:{self.bundle.spec_id.split('-')[-1]}"
                    )
                    inc.save(ep_id, key, t.result, by_ep.get(ep_id, []))
        llm_tests = sum(1 for vt in tests if vt.tier != "T0" and not vt.reused)
        t0_tests = sum(1 for vt in tests if vt.tier == "T0")
        reused = sum(1 for vt in tests if vt.reused)
        avg_llm = (
            (
                sum(r["actual_tokens"] for r in self.delegation.rows if r["start_tier"] != "T0")
                / llm_tests
            )
            if llm_tests
            else 3500
        )
        self.tokens_saved_estimate = int((t0_tests + reused) * avg_llm)
        if self.tokens_saved_estimate:
            metrics.inc("agentqa_tokens_saved_estimate_total", self.tokens_saved_estimate)
        if reused:
            self.incremental_stats["reused_tests"] = reused
            metrics.gauge("agentqa_incremental_reuse_ratio", reused / max(1, len(tests)))
        hall = dict(self.hallucination)
        hall["rate"] = hall["with_violations"] / hall["checked"] if hall["checked"] else 0.0
        if hall["checked"]:
            metrics.gauge("agentqa_eval_hallucination_rate", hall["rate"])
        report = RunReport(
            run_id=self.run_id,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            profile=self.cfg.profile,
            strategy=self.strategy.name,
            simulated=self.cfg.profile == "simulated",
            api_title=self.bundle.title if self.bundle else "?",
            api_version=self.bundle.version if self.bundle else "",
            target=self.cfg.base_url,
            trace_id=trace_id,
            phoenix_url=f"{self.cfg.phoenix_base}/projects" if self.cfg.phoenix_base else None,
            endpoints=len(self.bundle.endpoints) if self.bundle else 0,
            intents=list(intents.values()),
            results=run.results if run else [],
            tiers={vt.intent.id: vt.tier for vt in tests},
            findings=findings,
            uncovered=self.uncovered,
            guardrail_events=run.guardrail_events if run else [],
            quarantined_chunks=[c.id for c in self.bundle.chunks if c.quarantined]
            if self.bundle
            else [],
            hallucination=hall,
            cost=self.ledger.summary(),
            delegation={
                **self.delegation.summary(),
                "tokens_saved_estimate": self.tokens_saved_estimate,
            },
            aborted=aborted,
            incremental=self.incremental_stats,
            lessons_used=self.lessons_used,
        )
        for risk, row in report.coverage_by_risk().items():
            metrics.gauge(
                "agentqa_coverage_ratio", row["covered"] / max(1, row["planned"]), risk=risk
            )
        if self.cfg.judge and findings and not aborted:
            try:
                report.judge = judge_report(
                    report,
                    self.router.for_role("judge"),
                    only_ambiguous=self.on("cheap_verifiers_first"),
                )
            except (KillSwitchEngaged, BudgetExceeded) as exc:
                report.aborted = report.aborted or f"{type(exc).__name__}: {exc}"
            report.cost = self.ledger.summary()  # include the judge's tokens
        if self.lessons and not aborted:
            derive_lessons(self.lessons, self.run_id, self.rejections, findings)
        return report
