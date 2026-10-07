"""Budget guard: hard caps per run on tokens, list-equivalent USD, wall-clock and agent steps.

What it stops: a run that would blow past its budget. The check runs *before* each LLM call
(using the call's estimated tokens) and between pipeline steps; on breach the run aborts
cleanly with a partial report and the abort is a span event plus a metric.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from agentqa import config
from agentqa.llm.pricing import CostLedger
from agentqa.llm.types import BudgetExceeded, CallMetadata, LLMResult
from agentqa.obs import metrics, tracing


@dataclass
class BudgetGuard:
    ledger: CostLedger
    max_tokens: int
    max_usd: float
    max_wall_clock_s: float
    max_steps: int
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    tripped: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_config(cls, ledger: CostLedger, **overrides: float) -> BudgetGuard:
        b = config.guardrails().budget
        return cls(
            ledger=ledger,
            max_tokens=int(overrides.get("max_tokens", b.max_tokens)),
            max_usd=float(overrides.get("max_usd", b.max_usd_list_equivalent)),
            max_wall_clock_s=float(overrides.get("max_wall_clock_s", b.max_wall_clock_s)),
            max_steps=int(overrides.get("max_steps", b.max_agent_steps)),
        )

    # remaining budget, used by budget-aware allocation
    @property
    def tokens_used(self) -> int:
        return self.ledger.usage.total

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    def _trip(self, reason: str) -> None:
        self.tripped = reason
        metrics.inc("agentqa_guardrail_events_total", guardrail="budget", action="abort")
        tracing.event("guardrail.budget", action="abort", reason=reason)
        raise BudgetExceeded(reason)

    def check(self, est_tokens: int = 0) -> None:
        if self.tripped:
            raise BudgetExceeded(self.tripped)
        used = self.tokens_used
        if used + est_tokens > self.max_tokens:
            self._trip(f"token budget: {used}+{est_tokens} > {self.max_tokens}")
        if self.ledger.cost_list > self.max_usd:
            self._trip(f"usd budget: {self.ledger.cost_list:.4f} > {self.max_usd}")
        elapsed = time.monotonic() - self.started
        if elapsed > self.max_wall_clock_s:
            self._trip(f"wall clock: {elapsed:.0f}s > {self.max_wall_clock_s}s")

    def step(self, name: str) -> None:
        with self._lock:
            self.steps += 1
            steps = self.steps
        if steps > self.max_steps:
            self._trip(f"agent steps: {steps} > {self.max_steps} at {name}")
        self.check()

    # LLM router hooks
    def pre_call(self, meta: CallMetadata, est_tokens: int) -> None:
        self.check(est_tokens)

    def post_call(self, meta: CallMetadata, result: LLMResult) -> None:
        # Overshoot by one call is possible; the next pre_call stops the run.
        pass
