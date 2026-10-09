"""Generation stages of a run: ingest, Tier 0, planning, dispatch and test generation."""

from __future__ import annotations

import json
from typing import Any

from agentqa.agents.generator import Generator
from agentqa.agents.planner import Planner
from agentqa.agents.synthesizer import Synthesizer
from agentqa.dispatch import savings
from agentqa.dispatch.cascade import (
    CascadeOutcome,
    run_cascade,
)
from agentqa.dispatch.features import difficulty, features
from agentqa.dispatch.policy import Decision, choose_tier
from agentqa.guards import grounding, injection
from agentqa.guards.static_checks import check_code
from agentqa.ingest.ingestor import ingest
from agentqa.llm.types import LLMError
from agentqa.memory import LessonStore
from agentqa.models import CATEGORIES, SpecBundle, TestIntent, ValidatedTest
from agentqa.obs import tracing
from agentqa.orchestrator.graph import TaskGraph
from agentqa.orchestrator.state import PipelineState


class GenerationStages(PipelineState):
    """Ingest, Tier 0, plan (or reuse), dispatch and generation tasks."""

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
