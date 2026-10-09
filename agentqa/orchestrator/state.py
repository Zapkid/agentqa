"""Run-wide state and small helpers shared by the pipeline stages."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from agentqa import config
from agentqa.agents.reporter import Uncovered
from agentqa.config import agentqa_home
from agentqa.dispatch import strategies
from agentqa.dispatch.cascade import DelegationLedger, Verifier
from agentqa.dispatch.learned import RoutingStats
from agentqa.executor.runner import Executor
from agentqa.guards import killswitch
from agentqa.guards.budget import BudgetGuard
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.cache import DiskCache
from agentqa.llm.pricing import CostLedger
from agentqa.llm.router import ModelRouter
from agentqa.memory import LessonStore
from agentqa.models import SpecBundle, TestIntent, ValidatedTest
from agentqa.orchestrator.run_config import RunConfig
from agentqa.store import Store


class PipelineState:
    """Everything a run shares: configuration, clients, stores and run-level counters."""

    verifier: Verifier  # created by Pipeline.run() once the spec bundle exists

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
