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
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentqa import config
from agentqa.agents.generator import Generator
from agentqa.agents.planner import Planner
from agentqa.agents.reporter import RunReport, Uncovered, judge_report, render
from agentqa.agents.synthesizer import Synthesizer
from agentqa.agents.triage import Triage
from agentqa.config import agentqa_home
from agentqa.dispatch import savings, strategies
from agentqa.dispatch.cascade import (
    CascadeOutcome,
    DelegationLedger,
    ReferenceRunner,
    Verifier,
    run_cascade,
)
from agentqa.dispatch.features import difficulty, features
from agentqa.dispatch.learned import RoutingStats
from agentqa.dispatch.policy import Decision, choose_tier
from agentqa.executor.runner import Executor
from agentqa.guards import grounding, injection, killswitch
from agentqa.guards.budget import BudgetGuard
from agentqa.guards.static_checks import check_code
from agentqa.ingest.ingestor import ingest
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.cache import DiskCache
from agentqa.llm.pricing import CostLedger
from agentqa.llm.router import ModelRouter
from agentqa.llm.types import BudgetExceeded, KillSwitchEngaged, LLMError
from agentqa.memory import LessonStore, derive_lessons
from agentqa.models import CATEGORIES, Finding, RunResult, SpecBundle, TestIntent, ValidatedTest
from agentqa.obs import metrics, tracing
from agentqa.obs.logging import get_logger, run_id_var
from agentqa.orchestrator.graph import Supervisor, Task, TaskGraph
from agentqa.store import Store
from agentqa.target_config import TargetConfig

log = get_logger("agentqa.pipeline")


@dataclass
class RunConfig:
    spec: str | Path
    docs: str | Path | None
    target: TargetConfig
    base_url: str
    reference_url: str | None = None
    profile: str = "simulated"
    strategy: str = "S3"
    mechanism_overrides: dict[str, bool] = field(default_factory=dict)
    run_id: str | None = None
    out_root: Path | None = None
    budget: dict[str, float] = field(default_factory=dict)
    cache_mode: str | None = None
    use_lessons: bool = True
    server_log: Callable[[float], list[str]] | None = None
    store: Store | None = None
    vector_store: VectorStore | None = None
    concurrency: int | None = None
    judge: bool = True
    phoenix_base: str | None = "http://localhost:6006"


@dataclass
class RunOutput:
    run_id: str
    run_dir: Path
    report: RunReport
    paths: dict[str, Path]
    run_result: RunResult | None
    ledger: CostLedger
    delegation: DelegationLedger
    stats: dict[str, Any]
    tests: list[ValidatedTest] = field(default_factory=list)
    bundle: SpecBundle | None = None


class Pipeline:
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.strategy = strategies.get(cfg.strategy)
        mechanisms = {**self.strategy.mechanisms, **cfg.mechanism_overrides}
        self.dispatch = config.dispatch().with_overrides(**mechanisms)
        self.profile = config.profile(cfg.profile)
        self.run_id = (
            cfg.run_id
            or f"run-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        self.run_dir = (cfg.out_root or agentqa_home() / "runs") / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.store = cfg.store or Store()
        self.vstore = cfg.vector_store or VectorStore(agentqa_home() / "chroma")
        self.ledger = CostLedger()
        b = cfg.budget
        self.budget = BudgetGuard.from_config(self.ledger, **b)
        self.router = ModelRouter(
            cfg.profile,
            cache=DiskCache(mode=cfg.cache_mode) if cfg.cache_mode else DiskCache(),  # type: ignore[arg-type]
            ledger=self.ledger,
            pre_call=[killswitch.llm_hook, self.budget.pre_call],
            native_cache=self.dispatch.on("caching"),
        )
        self.delegation = DelegationLedger(self.store, self.run_id)
        self.routing = RoutingStats(self.store)
        self.executor = Executor(cfg.target)
        self.on = self.dispatch.on
        # run state (shared, written only by the task that owns each key)
        self.bundle: SpecBundle | None = None
        self.uncovered: list[Uncovered] = []
        self.rejections: list[tuple[str, str, str]] = []
        self.hallucination: dict[str, float] = {
            "checked": 0,
            "with_violations": 0,
            "quarantined": 0,
            "repaired": 0,
        }
        self.reused_tests: list[ValidatedTest] = []
        self.incremental_stats: dict[str, Any] = {}
        self.lessons: LessonStore | None = None
        self.lessons_used = 0
        self.tokens_saved_estimate = 0
        self.planner_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers

    def _tier(self, role: str) -> str:
        return self.strategy.forced_tier or self.profile.tier_for_role(role)

    def _tokens_for(self, task_id: str) -> int:
        return sum(e.usage.total for e in self.ledger.entries if e.task_id == task_id)

    def _cost_for(self, task_id: str) -> float:
        return sum(e.cost_usd_list_equivalent for e in self.ledger.entries if e.task_id == task_id)

    def _lessons_text(self, endpoint: str) -> str:
        if not self.lessons or not self.cfg.use_lessons:
            return ""
        items = self.lessons.for_endpoint(endpoint)
        self.lessons_used += len(items)
        return "\n".join(f"- {t}" for t in items)

    def _uncover(self, intent: TestIntent, reason: str) -> None:
        self.uncovered.append(
            Uncovered(
                intent_id=intent.id,
                endpoint=intent.endpoint,
                category=intent.category,
                risk=intent.risk,
                reason=reason,
            )
        )

    # ------------------------------------------------------------------ tasks

    def t_ingest(self) -> SpecBundle:
        classifier = injection.llm_classifier(self.router.for_role("injection_classifier"))
        bundle, counts = ingest(self.cfg.spec, self.cfg.docs, self.vstore, classifier=classifier)
        self.bundle = bundle
        self.lessons = LessonStore(self.store, api_key=bundle.title)
        self.verifier.endpoints = bundle.endpoints
        (self.run_dir / "ingest.json").write_text(json.dumps(counts), encoding="utf-8")
        return bundle

    def t_t0(self) -> list[ValidatedTest]:
        assert self.bundle is not None
        out = []
        for intent, test in Synthesizer(self.bundle.endpoints).synthesize():
            violations = check_code(test.code, test.test_name) + grounding.check_test(
                test, self.bundle.endpoints
            )
            if violations:  # a synthesizer bug; never ship it
                self._uncover(intent, f"T0 test failed its own checks: {violations[0].detail}")
                continue
            out.append(ValidatedTest(intent=intent, test=test, tier="T0"))
            self.delegation.record(
                task_id=intent.id,
                task_type=intent.category,
                features={},
                start_tier="T0",
                reason="deterministic_first: derivable from the spec",
                est_tokens=0,
                actual_tokens=0,
                cost_usd=0.0,
                verifier_outcome="static+grounding ok",
                escalation_path=["T0:ok"],
                final_result="validated",
            )
        return out

    def t_plan(self, endpoint_id: str) -> list[TestIntent]:
        assert self.bundle is not None
        ep = self.bundle.endpoint(endpoint_id)
        params = self.dispatch.params
        top_k, cap = (
            (params.retrieval_top_k, params.retrieval_token_cap * 4)
            if self.on("context_budgeting")
            else (10, 100_000)
        )
        planner = Planner(
            self.bundle,
            self.vstore,
            top_k=top_k,
            char_cap=cap,
            max_intents=params.max_intents_per_endpoint,
        )
        intents, stats = planner.plan(
            ep,
            self.router.for_tier(self._tier("planner")),
            list(CATEGORIES),
            lessons=self._lessons_text(endpoint_id),
            task_id=f"plan:{endpoint_id}",
            spec_derived_covered=self.on("deterministic_first"),
        )
        self.planner_stats[endpoint_id] = stats.__dict__
        return intents

    def t_reuse(self, endpoint_id: str, stored: dict[str, Any]) -> list[TestIntent]:
        intents = [TestIntent.model_validate(i) for i in stored["intents"]]
        for t in stored["tests"]:
            vt = ValidatedTest.model_validate(t)
            self.reused_tests.append(vt.model_copy(update={"reused": True}))
        return intents

    def t_generate(
        self,
        intent: TestIntent,
        decision: Decision,
        feats: dict[str, Any],
        prior: list[Any] | None = None,
    ) -> ValidatedTest | None:
        """``prior``: violations of this intent's batched first attempt (counted as pre-repair)."""
        assert self.bundle is not None
        gen = Generator(self.bundle, self.vstore)
        lessons = self._lessons_text(intent.endpoint)

        def generate(it: TestIntent, tier: str, feedback: str | None) -> Any:
            return gen.generate(
                it, self.router.for_tier(tier), feedback=feedback, lessons=lessons, task_id=it.id
            )

        outcome: CascadeOutcome = run_cascade(
            intent,
            decision.tier,
            decision.can_escalate,
            generate,
            self.verifier,
            lambda: self._tokens_for(intent.id),
        )
        if prior:
            outcome.pre_repair = prior
            outcome.path.insert(0, f"T1:batch:{prior[0].kind}")
            if outcome.validated is not None:
                outcome.validated.repaired = True
        self._account(intent, decision, feats, outcome)
        return outcome.validated

    def _account(
        self,
        intent: TestIntent,
        decision: Decision,
        feats: dict[str, Any],
        outcome: CascadeOutcome,
        est_tokens: int = savings.EST_TOKENS["T1"],
    ) -> None:
        self.hallucination["checked"] += 1
        if outcome.hallucinated:
            self.hallucination["with_violations"] += 1
        for v in outcome.pre_repair:
            self.rejections.append((intent.endpoint, v.kind, v.detail))
        if outcome.validated is None:
            self.hallucination["quarantined"] += 1
            self._uncover(intent, outcome.uncovered_reason or "generation failed")
        elif outcome.validated.repaired:
            self.hallucination["repaired"] += 1
        escalated = any(p.startswith("T2") for p in outcome.path) and decision.tier == "T1"
        if self.on("learned_routing"):
            if decision.tier == "T1":
                self.routing.update(
                    intent.category,
                    "T1",
                    success=not escalated and outcome.validated is not None,
                    tokens=outcome.tokens,
                )
            if escalated or decision.tier == "T2":
                self.routing.update(
                    intent.category,
                    "T2",
                    success=outcome.validated is not None,
                    tokens=outcome.tokens,
                )
        self.delegation.record(
            task_id=intent.id,
            task_type=intent.category,
            features={**feats, "difficulty": difficulty(feats)},
            start_tier=decision.tier,
            reason=decision.reason,
            est_tokens=est_tokens,
            actual_tokens=outcome.tokens,
            cost_usd=round(self._cost_for(intent.id), 6),
            verifier_outcome=(outcome.pre_repair[0].kind if outcome.pre_repair else "ok"),
            escalation_path=outcome.path,
            final_result="validated" if outcome.validated else "quarantined",
        )

    def t_batch(
        self,
        intents: list[TestIntent],
        decisions: dict[str, Decision],
        feats: dict[str, dict[str, Any]],
    ) -> list[ValidatedTest | None]:
        """One call for several easy intents; items that fail verification fall back to the cascade."""
        assert self.bundle is not None
        gen = Generator(self.bundle, self.vstore)
        task_id = f"batch:{intents[0].id}"
        try:
            items = gen.generate_batch(intents, self.router.for_tier("T1"), task_id=task_id)
        except LLMError as exc:
            items = {i.id: f"batch failed: {exc}" for i in intents}
        share = self._tokens_for(task_id) // max(1, len(intents))
        out: list[ValidatedTest | None] = []
        for intent in intents:
            item = items[intent.id]
            violations = (
                self.verifier.verify(intent, item, "T1")
                if not isinstance(item, str)
                else [grounding.GuardViolation(kind="generation_error", detail=item)]
            )
            if not violations and not isinstance(item, str):
                outcome = CascadeOutcome(
                    validated=ValidatedTest(intent=intent, test=item, tier="T1"),
                    path=["T1:batch:ok"],
                    tokens=share,
                )
                self._account(intent, decisions[intent.id], feats[intent.id], outcome)
                out.append(outcome.validated)
                continue
            tracing.event("batch.item_rejected", intent=intent.id, reason=violations[0].kind)
            out.append(
                self.t_generate(intent, decisions[intent.id], feats[intent.id], prior=violations)
            )
        return out

    # ------------------------------------------------------------------ dispatch (fan-in of planning)

    def t_dispatch(self, graph: TaskGraph) -> dict[str, Any]:
        assert self.bundle is not None
        planned: list[TestIntent] = []
        for t in graph.of_kind("plan") + graph.of_kind("reuse"):
            if t.status == "done":
                planned.extend(t.result or [])
            elif t.status == "failed":
                tracing.event("plan.failed", task=t.id, error=t.error)
        reused_ids = {vt.intent.id for vt in self.reused_tests}
        todo = [i for i in planned if i.id not in reused_ids]
        removed: list[TestIntent] = []
        if self.on("dedupe"):
            todo, removed = savings.dedupe(todo, self.dispatch.params.dedupe_similarity)
        feats = {i.id: features(i, self._endpoint(i.endpoint)) for i in todo}
        diffs = {i.id: difficulty(f) for i, f in zip(todo, feats.values(), strict=True)}
        decisions = {
            i.id: choose_tier(
                i.id,
                i.category,
                diffs[i.id],
                self.dispatch,
                self.profile,
                self.routing if self.on("learned_routing") else None,
                self.strategy.forced_tier,
            )
            for i in todo
        }
        allocation_skipped: dict[str, str] = {}
        if self.on("budget_allocation"):
            alloc = savings.allocate(todo, int(self.budget.tokens_remaining * 0.7))
            allocation_skipped = alloc.skipped
            for iid, cap in alloc.tier_cap.items():
                if cap == "T1" and decisions[iid].can_escalate:
                    decisions[iid] = Decision(
                        decisions[iid].tier,
                        decisions[iid].reason + "; budget: no escalation",
                        False,
                    )
        for i in todo:
            if i.id in allocation_skipped:
                self._uncover(i, allocation_skipped[i.id])
        todo = [i for i in todo if i.id not in allocation_skipped]
        batch_groups: list[list[TestIntent]] = []
        singles = todo
        if self.on("batching"):
            eligible = {i.id: diffs[i.id] for i in todo if decisions[i.id].tier == "T1"}
            batch_groups, singles = savings.batches(
                [i for i in todo if i.id in eligible], eligible, self.dispatch.params.batch_size
            )
            singles += [i for i in todo if i.id not in eligible]
        return {
            "planned": planned,
            "removed": removed,
            "decisions": decisions,
            "features": feats,
            "batches": batch_groups,
            "singles": singles,
        }

    def _endpoint(self, endpoint_id: str) -> Any:
        assert self.bundle is not None
        try:
            return self.bundle.endpoint(endpoint_id)
        except KeyError:
            return None

    # ------------------------------------------------------------------ graph

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
