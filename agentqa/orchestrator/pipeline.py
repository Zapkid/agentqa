"""The AgentQA run: spec + docs -> plan -> tests -> execution -> triage -> report.

Task graph (built by spawns as results arrive):

    ingest ─┬─ t0 (Tier 0, 0 tokens)                      ┐
            ├─ plan:<endpoint> ... (or reuse:<endpoint>)   ├─ dispatch ─┬─ gen:<intent> ... ┐
            │                                              ┘            └─ batch:<n> ...    ├─ execute ─ triage
                                                                                            ┘
The report is rendered after the graph drains, so every run (including an aborted one) ends
with a report. Agents never call each other; they only see the artifacts passed to them here.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentqa.agents.reporter import render
from agentqa.agents.triage import Triage
from agentqa.dispatch import savings
from agentqa.dispatch.cascade import (
    ReferenceRunner,
    Verifier,
)
from agentqa.executor.runner import Executor
from agentqa.guards import killswitch
from agentqa.memory import LessonStore
from agentqa.models import Finding, RunResult, SpecBundle, TestIntent, ValidatedTest
from agentqa.obs import metrics, tracing
from agentqa.obs.logging import run_id_var
from agentqa.orchestrator.graph import Supervisor, Task, TaskGraph
from agentqa.orchestrator.reporting import Reporting
from agentqa.orchestrator.run_config import RunConfig, RunOutput
from agentqa.orchestrator.stages import GenerationStages

__all__ = ["Pipeline", "RunConfig", "RunOutput"]


class Pipeline(GenerationStages, Reporting):
    """One run: builds the task graph, executes the suite, triages failures and reports."""

    def build(self, graph: TaskGraph) -> None:
        def spawn_after_ingest(bundle: SpecBundle) -> list[Task]:
            self.bundle = bundle
            self.lessons = LessonStore(self.store, api_key=bundle.title)
            children: list[Task] = []
            deps = []
            if self.on("deterministic_first"):
                children.append(
                    Task(
                        "t0",
                        "t0",
                        self.t_t0,
                        deps=["ingest"],
                        priority=100,
                        dump=lambda r: [v.model_dump() for v in r],
                        load=lambda p: [ValidatedTest.model_validate(v) for v in p],
                    )
                )
                deps.append("t0")
            inc = savings.Incremental(self.store, bundle.title)
            reused = 0
            for ep in bundle.endpoints:
                key = f"{bundle.endpoint_hashes[ep.id]}:{bundle.spec_id.split('-')[-1]}"
                stored = inc.load(ep.id, key) if self.on("incremental") else None
                if stored is not None:
                    reused += 1
                    children.append(
                        Task(
                            f"reuse:{ep.id}",
                            "reuse",
                            lambda e=ep.id, s=stored: self.t_reuse(e, s),  # type: ignore[misc]
                            deps=["ingest"],
                            priority=90,
                        )
                    )
                else:
                    children.append(
                        Task(
                            f"plan:{ep.id}",
                            "plan",
                            lambda e=ep.id: self.t_plan(e),  # type: ignore[misc]
                            deps=["ingest"],
                            priority=80,
                            dump=lambda r: [i.model_dump() for i in r],
                            load=lambda p: [TestIntent.model_validate(i) for i in p],
                        )
                    )
                deps.append(children[-1].id)
            self.incremental_stats = {
                "endpoints": len(bundle.endpoints),
                "reused_endpoints": reused,
            }
            children.append(
                Task(
                    "dispatch",
                    "dispatch",
                    lambda: self.t_dispatch(graph),
                    deps=deps,
                    tolerate_failures=True,
                    priority=70,
                    spawn=spawn_after_dispatch,
                    dump=lambda r: None,
                    resumable=False,
                )
            )
            return children

        def spawn_after_dispatch(d: dict[str, Any]) -> list[Task]:
            children: list[Task] = []
            gen_ids = ["t0"] if graph.tasks.get("t0") else []
            for n, group in enumerate(d["batches"]):
                tid = f"batch:{n}"
                children.append(
                    Task(
                        tid,
                        "batch",
                        lambda g=group: self.t_batch(g, d["decisions"], d["features"]),  # type: ignore[misc]
                        deps=["dispatch"],
                        priority=max(i.risk for i in group) * 10,
                        dump=lambda r: [v.model_dump() if v else None for v in r],
                        load=lambda p: [ValidatedTest.model_validate(v) if v else None for v in p],
                    )
                )
                gen_ids.append(tid)
            for intent in d["singles"]:
                tid = f"gen:{intent.id}"
                children.append(
                    Task(
                        tid,
                        "gen",
                        lambda i=intent: self.t_generate(  # type: ignore[misc]
                            i, d["decisions"][i.id], d["features"][i.id]
                        ),
                        deps=["dispatch"],
                        priority=intent.risk * 10,
                        dump=lambda r: r.model_dump() if r else None,
                        load=lambda p: ValidatedTest.model_validate(p) if p else None,
                    )
                )
                gen_ids.append(tid)
            children.append(
                Task(
                    "execute",
                    "execute",
                    lambda: self.t_execute(graph),
                    deps=gen_ids,
                    tolerate_failures=True,
                    priority=10,
                    spawn=lambda r: [triage_task],
                    dump=lambda r: r.model_dump(),
                    load=RunResult.model_validate,
                )
            )
            return children

        triage_task = Task(
            "triage",
            "triage",
            lambda: self.t_triage(graph),
            deps=["execute"],
            priority=5,
            dump=lambda r: [f.model_dump() for f in r],
            load=lambda p: [Finding.model_validate(f) for f in p],
        )
        graph.add(
            Task(
                "ingest",
                "ingest",
                self.t_ingest,
                priority=1000,
                spawn=spawn_after_ingest,
                dump=lambda r: r.model_dump(),
                load=SpecBundle.model_validate,
            )
        )

    def t_execute(self, graph: TaskGraph) -> RunResult:
        tests = self.validated_tests(graph)
        killswitch.check("before execute")
        return self.executor.run(
            tests, self.cfg.base_url, self.run_dir, server_log=self.cfg.server_log
        )

    def t_triage(self, graph: TaskGraph) -> list[Finding]:
        run: RunResult = graph.result("execute")
        tests = self.validated_tests(graph)
        intents = {vt.intent.id: vt.intent for vt in tests}
        assert self.bundle is not None
        triage = Triage(
            self.bundle,
            escalate_confidence=self.dispatch.params.triage_escalate_confidence,
            cluster_first=self.on("cluster_first_triage"),
        )
        forced = self.strategy.forced_tier
        cascade = self.on("cascade")

        def client_for(tier: str) -> Any:
            if forced:
                return self.router.for_tier(forced)
            if not cascade:
                return self.router.for_tier(self.profile.tier_for_role("triage"))
            return self.router.for_tier(tier)

        if not cascade or forced:
            triage.escalate_confidence = -1.0  # no escalation without the cascade mechanism
        findings = triage.triage(run, intents, client_for)
        self.triage_stats = triage.stats
        return findings

    # ------------------------------------------------------------------ run

    def run(self) -> RunOutput:
        token = run_id_var.set(self.run_id)
        started = time.monotonic()
        self.store.save_run(
            self.run_id,
            profile=self.cfg.profile,
            strategy=self.strategy.name,
            status="running",
            run_dir=str(self.run_dir),
        )
        killswitch.check("run start")
        graph = TaskGraph()
        with tracing.span(
            f"run {self.run_id}",
            "run",
            **{
                "agentqa.run_id": self.run_id,
                "agentqa.profile": self.cfg.profile,
                "agentqa.strategy": self.strategy.name,
            },
        ) as root:
            trace_id = format(root.get_span_context().trace_id, "032x")
            self.verifier = Verifier(
                self.bundle.endpoints if self.bundle else [],
                ReferenceRunner(
                    Executor(self.cfg.target, reruns=0),
                    self.cfg.reference_url,
                    self.run_dir / "verify",
                )
                if self.cfg.reference_url
                else None,
            )
            self.build(graph)

            def before(task: Task) -> None:
                killswitch.check(f"before {task.id}")
                self.budget.step(task.id)

            sup = Supervisor(
                graph,
                concurrency=self.cfg.concurrency or self.dispatch.params.concurrency,
                store=self.store,
                run_id=self.run_id,
                before_task=before,
            )
            sup.run()
            report = self.finish(graph, sup.aborted, trace_id)
            root.set_attribute("agentqa.findings", len(report.findings))
            root.set_attribute("agentqa.cost_usd_list_equivalent", self.ledger.cost_list)
        elapsed = time.monotonic() - started
        metrics.observe(
            "agentqa_run_duration_seconds",
            elapsed,
            profile=self.cfg.profile,
            strategy=self.strategy.name,
        )
        tracing.flush()
        n_spans = tracing.dump_spans(self.run_dir / "spans.jsonl", trace_id)
        paths = render(report, self.run_dir)
        stats = {
            "elapsed_s": round(elapsed, 2),
            "spans": n_spans,
            "triage": getattr(self, "triage_stats", {}),
            "planner": self.planner_stats,
            "hallucination": self.hallucination,
            "routing_table": self.routing.table(),
            "graph": {t.id: t.status for t in graph.tasks.values()},
        }
        (self.run_dir / "stats.json").write_text(
            json.dumps(stats, indent=1, default=str), encoding="utf-8"
        )
        self.store.save_run(
            self.run_id,
            status="aborted" if report.aborted else "done",
            trace_id=trace_id,
            spec_id=self.bundle.spec_id if self.bundle else None,
            api_title=report.api_title,
            summary=json.dumps(
                {
                    "outcomes": report.outcome_counts(),
                    "findings": len(report.findings),
                    "product_bugs": len(report.product_bugs()),
                    "cost": self.ledger.summary()["cost_usd_list_equivalent"],
                    "tokens": self.ledger.usage.total,
                }
            ),
        )
        run_id_var.reset(token)
        return RunOutput(
            self.run_id,
            self.run_dir,
            report,
            paths,
            graph.result("execute"),
            self.ledger,
            self.delegation,
            stats,
            self.validated_tests(graph),
            self.bundle,
        )
